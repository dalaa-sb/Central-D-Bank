import math
import json
import os
import requests
import logging
from decimal import Decimal, getcontext
from typing import Optional
from cryptography.fernet import Fernet
from sklearn.ensemble import IsolationForest
import numpy as np
import bcrypt
from sqlalchemy import create_engine, Column, String, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from ratelimit import limits, sleep_and_retry
import openai
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import getpass

# Set decimal precision for money calculations
getcontext().prec = 10

# Configure OpenAI API key from environment
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up SQLite database
engine = create_engine('sqlite:///bank.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)

# Updated base class for SQLAlchemy 2.0
class Base(DeclarativeBase):
    pass

# Database models with encryption for sensitive data
class AccountRecord(Base):
    __tablename__ = 'accounts'
    id: Mapped[int] = mapped_column(primary_key=True)
    owner: Mapped[str] = mapped_column(String)
    encrypted_balance: Mapped[str] = mapped_column(Text)
    base_currency: Mapped[str] = mapped_column(String)
    hashed_pin: Mapped[str] = mapped_column(Text)

class TransactionLog(Base):
    __tablename__ = 'logs'
    id: Mapped[int] = mapped_column(primary_key=True)
    encrypted_message: Mapped[str] = mapped_column(Text)

# Create all tables
Base.metadata.create_all(engine)

# Rate limiting: 5 calls per 60 seconds
@sleep_and_retry
@limits(calls=5, period=60)
def rate_limited_action():
    pass

class Currency:
    SUPPORTED = ['USD', 'EUR', 'GBP', 'AED']
    CURRENCY_NAMES = {
        'USD': 'US Dollar',
        'EUR': 'Euro',
        'GBP': 'British Pound',
        'AED': 'UAE Dirham'
    }

    @staticmethod
    def is_valid(currency: str) -> bool:
        return currency.upper() in Currency.SUPPORTED

    @staticmethod
    def get_name(currency: str) -> str:
        return Currency.CURRENCY_NAMES.get(currency.upper(), currency)

    @staticmethod
    def get_exchange_rate_to_usd(currency: str) -> float:
        """Get exchange rate from given currency to USD"""
        rates = {
            'USD': 1.0,
            'EUR': 1.18,  # 1 EUR = 1.18 USD
            'GBP': 1.37,  # 1 GBP = 1.37 USD
            'AED': 0.27   # 1 AED = 0.27 USD
        }
        return rates.get(currency.upper(), 1.0)

    @staticmethod
    def convert_to_usd(amount: float, from_currency: str) -> float:
        """Convert amount from given currency to USD"""
        rate = Currency.get_exchange_rate_to_usd(from_currency)
        return amount * rate

class BankAccount:
    def __init__(self, owner: str, balance: float, base_currency: str = 'USD', pin: str = ''):
        self._owner = owner.strip()
        if not self._owner or not is_valid_name(self._owner):
            raise ValueError('Invalid owner name.')
        if balance < 0 or not math.isfinite(balance):
            raise ValueError('Balance must be non-negative and finite.')
        self._balance = Decimal(str(balance))
        self._base_currency = base_currency.upper()
        if not Currency.is_valid(self._base_currency):
            raise ValueError('Invalid base currency.')
        self._hashed_pin = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()) if pin else None
        self._save_to_db()

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def balance(self) -> Decimal:
        return self._load_balance_from_db()

    @property
    def base_currency(self) -> str:
        return self._base_currency

    def _save_to_db(self):
        session = Session()
        try:
            record = session.query(AccountRecord).filter_by(owner=self._owner).first()
            if not record:
                record = AccountRecord(
                    owner=self._owner,
                    base_currency=self._base_currency,
                    hashed_pin=self._hashed_pin.decode() if self._hashed_pin else None
                )
            key = os.getenv('ENCRYPTION_KEY')
            if not key:
                raise ValueError("ENCRYPTION_KEY not set in environment variables")
            cipher = Fernet(key.encode())
            record.encrypted_balance = cipher.encrypt(str(self._balance).encode()).decode()
            if self._hashed_pin and not record.hashed_pin:
                record.hashed_pin = self._hashed_pin.decode()
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database error in _save_to_db: {e}")
            raise
        finally:
            session.close()

    def _load_balance_from_db(self) -> Decimal:
        session = Session()
        try:
            record = session.query(AccountRecord).filter_by(owner=self._owner).first()
            if record and record.encrypted_balance:
                key = os.getenv('ENCRYPTION_KEY')
                if not key:
                    raise ValueError("ENCRYPTION_KEY not set in environment variables")
                cipher = Fernet(key.encode())
                decrypted = cipher.decrypt(record.encrypted_balance.encode()).decode()
                return Decimal(decrypted)
            return Decimal('0')
        finally:
            session.close()

    def authenticate_pin(self, pin: str) -> bool:
        session = Session()
        try:
            record = session.query(AccountRecord).filter_by(owner=self._owner).first()
            if record and record.hashed_pin:
                try:
                    return bcrypt.checkpw(pin.encode(), record.hashed_pin.encode())
                except ValueError:
                    return False
            return False
        finally:
            session.close()

    def _convert_to_base(self, amount: Decimal, from_currency: str) -> Decimal:
        if from_currency.upper() == self._base_currency:
            return amount
        rate = get_exchange_rate(from_currency.upper(), self._base_currency)
        return amount * Decimal(str(rate))

    def _detect_anomaly(self, amount: float, currency: str) -> str:
        config_path = os.path.join(os.getcwd(), 'config.json')
        alert = ''
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            data = np.array(config['anomaly_model_data'])
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(data)
            currency_mapping = {'USD': 0, 'EUR': 1, 'GBP': 2, 'AED': 3}
            currency_index = currency_mapping.get(currency.upper(), 0)
            if model.predict([[amount, currency_index]])[0] == -1:
                alert = ' ⚠️ Suspicious transaction detected.⚠️'
                if openai.api_key:
                    try:
                        response = openai.Completion.create(
                            model="text-davinci-003",
                            prompt=f"Explain why a {amount} {currency} transaction might be fraudulent.",
                            max_tokens=50
                        )
                        alert += f" AI Insight: {response.choices[0].text.strip()}"
                    except Exception as e:
                        alert += f" (AI insight unavailable: {str(e)})"
        except Exception:
            # Silently fail - anomaly detection is optional
            pass
        return alert

    def _get_recommendation(self) -> str:
        if not openai.api_key:
            return ''
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Suggest a banking action for user {self._owner} with balance {self.balance} in {self._base_currency}.",
                max_tokens=50
            )
            return f" AI Recommendation: {response.choices[0].text.strip()}🤖"
        except Exception:
            return ''

    def _auto_suggest_transfer(self, other: 'BankAccount') -> str:
        config_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            class BankingEnv(gym.Env):
                def __init__(self):
                    super(BankingEnv, self).__init__()
                    self.action_space = gym.spaces.Discrete(2)
                    self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    return np.array([0.1, 0.5], dtype=np.float32), {}

                def step(self, action):
                    observation = np.array([0.1, 0.5], dtype=np.float32)
                    reward = 1.0 if action == 1 else 0.0
                    terminated = True
                    truncated = False
                    info = {}
                    return observation, reward, terminated, truncated, info

                def render(self):
                    pass

                def close(self):
                    pass

            env = DummyVecEnv([lambda: BankingEnv()])
            model = PPO('MlpPolicy', env, verbose=0)
            model.learn(total_timesteps=100)
            action, _ = model.predict(np.array([[0.1, 0.5]]))
            if action[0] == 1:
                return f" AI Suggests: Transfer 10% of balance to {other.owner}.🤖"
        except Exception:
            # Silently fail - RL suggestion is optional
            pass
        return ''

    def deposit(self, amount: float, currency: str):
        rate_limited_action()
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError('Deposit amount must be positive and finite.')
        if not Currency.is_valid(currency):
            raise ValueError('Invalid currency.')
        alert = self._detect_anomaly(amount, currency)
        if alert:
            print(f'\n{alert}')
        converted_amount = self._convert_to_base(Decimal(str(amount)), currency)
        current_balance = self.balance
        if current_balance + converted_amount > Decimal('1000000000'):
            raise ValueError('Deposit would exceed maximum balance.')
        self._balance = current_balance + converted_amount
        self._save_to_db()
        self._log_transaction(f'Deposit: {amount:.2f} {currency} by {self._owner}')
        usd_equivalent = Currency.convert_to_usd(float(converted_amount), self._base_currency)
        print(f'\n Deposited {amount:.2f} {currency.upper()} ({Currency.get_name(currency)})✅')
        print(f'   Converted to: {converted_amount:.2f} {self._base_currency} ({Currency.get_name(self._base_currency)})')
        print(f'   USD Equivalent: ${usd_equivalent:.2f}')
        print(f'   New balance: {self._balance:.2f} {self._base_currency}')
        recommendation = self._get_recommendation()
        if recommendation:
            print(f'   {recommendation}')

    def withdraw(self, amount: float, currency: str):
        rate_limited_action()
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError('Withdrawal amount must be positive and finite.')
        if not Currency.is_valid(currency):
            raise ValueError('Invalid currency.')
        alert = self._detect_anomaly(amount, currency)
        if alert:
            print(f'\n{alert}')
        converted_amount = self._convert_to_base(Decimal(str(amount)), currency)
        current_balance = self.balance
        if converted_amount > current_balance:
            raise ValueError(f'Insufficient balance. Current balance: {current_balance:.2f} {self._base_currency}')
        self._balance = current_balance - converted_amount
        self._save_to_db()
        self._log_transaction(f'Withdrawal: {amount:.2f} {currency} by {self._owner}')
        usd_equivalent = Currency.convert_to_usd(float(converted_amount), self._base_currency)
        print(f'\nWithdrew {amount:.2f} {currency.upper()} ({Currency.get_name(currency)})✅ ')
        print(f'   Converted to: {converted_amount:.2f} {self._base_currency} ({Currency.get_name(self._base_currency)})')
        print(f'   USD Equivalent: ${usd_equivalent:.2f}')
        print(f'   New balance: {self._balance:.2f} {self._base_currency}')
        recommendation = self._get_recommendation()
        if recommendation:
            print(f'   {recommendation}')

    def show_balance(self):
        balance_usd = Currency.convert_to_usd(float(self.balance), self._base_currency)
        print(f'\n👤 Owner: {self._owner}')
        print(f'💰 Balance: {self.balance:.2f} {self._base_currency} ({Currency.get_name(self._base_currency)})')
        print(f'💵 USD Equivalent: ${balance_usd:.2f}')
        recommendation = self._get_recommendation()
        if recommendation:
            print(f'   {recommendation}')

    def transfer_to(self, other: 'BankAccount', amount: float, currency: str):
        rate_limited_action()
        if self is other:
            raise ValueError('Cannot transfer to the same account.')
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError('Transfer amount must be positive and finite.')
        if not Currency.is_valid(currency):
            raise ValueError('Invalid currency.')
        alert = self._detect_anomaly(amount, currency)
        if alert:
            print(f'\n{alert}')
        converted_sender = self._convert_to_base(Decimal(str(amount)), currency)
        current_balance = self.balance
        if converted_sender > current_balance:
            raise ValueError(f'Insufficient balance for transfer. Current balance: {current_balance:.2f} {self._base_currency}')
        converted_receiver = converted_sender * Decimal(str(get_exchange_rate(self._base_currency, other._base_currency))) if self._base_currency != other._base_currency else converted_sender
        other_current_balance = other.balance
        if other_current_balance + converted_receiver > Decimal('1000000000'):
            raise ValueError('Transfer would exceed recipient\'s maximum balance.')
        self._balance = current_balance - converted_sender
        other._balance = other_current_balance + converted_receiver
        self._save_to_db()
        other._save_to_db()
        self._log_transaction(f'Transfer: {amount:.2f} {currency} from {self._owner} to {other._owner}')

        sender_usd = Currency.convert_to_usd(float(converted_sender), self._base_currency)
        receiver_usd = Currency.convert_to_usd(float(converted_receiver), other._base_currency)

        print(f'\n {self._owner} transferred {amount:.2f} {currency.upper()} ({Currency.get_name(currency)})✅')
        print(f'   From: {converted_sender:.2f} {self._base_currency} (${sender_usd:.2f} USD)')
        print(f'   To: {converted_receiver:.2f} {other._base_currency} (${receiver_usd:.2f} USD)')
        print(f'\n   New balance for {self._owner}: {self._balance:.2f} {self._base_currency} (${Currency.convert_to_usd(float(self._balance), self._base_currency):.2f} USD)')
        print(f'   New balance for {other._owner}: {other._balance:.2f} {other._base_currency} (${Currency.convert_to_usd(float(other._balance), other._base_currency):.2f} USD)')

        suggestion = self._auto_suggest_transfer(other)
        if suggestion:
            print(f'\n   {suggestion}')

    def _log_transaction(self, message: str):
        session = Session()
        try:
            key = os.getenv('ENCRYPTION_KEY')
            if not key:
                raise ValueError("ENCRYPTION_KEY not set in environment variables")
            cipher = Fernet(key.encode())
            encrypted_message = cipher.encrypt(message.encode()).decode()
            log = TransactionLog(encrypted_message=encrypted_message)
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database error in _log_transaction: {e}")
            raise
        finally:
            session.close()

def is_valid_name(name: str) -> bool:
    trimmed = name.strip()
    if not trimmed:
        return False
    for ch in trimmed:
        is_upper = 'A' <= ch <= 'Z'
        is_lower = 'a' <= ch <= 'z'
        is_space = ch == ' '
        if not (is_upper or is_lower or is_space):
            return False
    return True

def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    try:
        response = requests.get(f'https://api.exchangerate-api.com/v4/latest/{from_currency}', timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['rates'].get(to_currency, 1.0)
    except (requests.RequestException, KeyError):
        config_path = os.path.join(os.getcwd(), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('exchange_rates', {}).get(f'{from_currency}_to_{to_currency}', 1.0)
        except (FileNotFoundError, json.JSONDecodeError):
            raise ValueError('Exchange rate unavailable. Please check API or config.')

def get_valid_name(prompt: str) -> str:
    while True:
        user_input = input(prompt).strip()
        if user_input and is_valid_name(user_input):
            return user_input
        print('\nPlease do not use numbers or symbols for names. Name must not be empty.')

def get_valid_balance(owner: str) -> float:
    while True:
        try:
            user_input = input(f'Enter starting balance for {owner}: ').strip()
            balance = float(user_input or '0')
            if balance >= 0 and math.isfinite(balance) and balance <= 1000000000.0:
                return balance
            else:
                print('\nStarting balance must be non-negative, finite, and within limits.')
        except ValueError:
            print('\nInvalid input. Please enter a valid number.')

def get_valid_currency(prompt: str) -> str:
    while True:
        print("\nAvailable currencies:")
        for curr in Currency.SUPPORTED:
            name = Currency.get_name(curr)
            to_usd = Currency.get_exchange_rate_to_usd(curr)
            print(f"  {curr} - {name} (1 {curr} = ${to_usd:.2f} USD)")

        user_input = input(prompt).strip().upper()
        if Currency.is_valid(user_input):
            return user_input
        print(f'\nInvalid currency. Supported: {", ".join(Currency.SUPPORTED)}')

def get_pin(owner: str) -> str:
    while True:
        try:
            pin = getpass.getpass(f'Enter PIN for {owner} (4 digits, hidden input): ')
            if len(pin) == 4 and pin.isdigit():
                return pin
            print('\nPIN must be exactly 4 digits.')
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            raise
        except Exception:
            pin = input(f'Enter PIN for {owner} (4 digits): ').strip()
            if len(pin) == 4 and pin.isdigit():
                return pin
            print('\nPIN must be exactly 4 digits.')

def get_hidden_pin(prompt: str) -> str:
    import sys

    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        return getpass.getpass(prompt)
    else:
        return getpass.getpass(prompt)

def main():
    print("Initializing database...")
    Base.metadata.create_all(engine)
    print("Database ready!")

    print('\n' + '='*50)
    print('      WELCOME TO CENTRAL D BANK')
    print('='*50)

    print("\n💱 Available Currencies with USD Exchange Rates:")
    for curr in Currency.SUPPORTED:
        name = Currency.get_name(curr)
        to_usd = Currency.get_exchange_rate_to_usd(curr)
        print(f"   {curr}: {name} | 1 {curr} = ${to_usd:.2f} USD")
    print()

    name1 = get_valid_name('Enter name for first account: ')
    base_currency1 = get_valid_currency(f'Choose base currency for {name1}: ')
    balance1 = get_valid_balance(name1)
    pin1 = get_pin(name1)
    if balance1 == 0:
        print(f'Note: The account for {name1} is empty.')

    try:
        account1 = BankAccount(name1, balance1, base_currency1, pin1)
        print(f" Account for {name1} created successfully!✅")
    except Exception as e:
        print(f" Error creating account for {name1}: {e}❌")
        return

    name2 = get_valid_name('\nEnter name for second account: ')
    base_currency2 = get_valid_currency(f'Choose base currency for {name2}: ')
    balance2 = get_valid_balance(name2)
    pin2 = get_pin(name2)
    if balance2 == 0:
        print(f'Note: The account for {name2} will start empty.')

    try:
        account2 = BankAccount(name2, balance2, base_currency2, pin2)
        print(f" Account for {name2} created successfully!✅")
    except Exception as e:
        print(f" Error creating account for {name2}: {e}❌")
        return

    print('\n' + '='*50)
    print('           ACCOUNTS CREATED')
    print('='*50)
    print('\nAccount 1:')
    account1.show_balance()
    print('\nAccount 2:')
    account2.show_balance()

    while True:
        print('\n' + '='*50)
        print('           MAIN MENU')
        print('='*50)
        print('\nChoose the account you want to use:')
        print(f'1) {account1.owner} ({account1.base_currency})')
        print(f'2) {account2.owner} ({account2.base_currency})')
        print('3) Exit')
        account_choice = input('\nChoose 1, 2, or 3: ').strip()

        if account_choice == '3':
            print('\n' + '='*50)
            print('  Thank you for choosing Central D Bank!')
            print('               Until next time. 👋')
            print('='*50)
            break

        current_account = None
        current_pin = ''
        if account_choice == '1':
            current_account = account1
            current_pin = pin1
        elif account_choice == '2':
            current_account = account2
            current_pin = pin2
        else:
            print('\n Invalid choice. Try again.❌')
            continue

        entered_pin = get_hidden_pin(f'Enter PIN for account of {current_account.owner}: ')
        if not current_account.authenticate_pin(entered_pin):
            print(' Invalid PIN. Access denied.❌')
            continue

        print(f'\n Access granted!✅')
        print(f'You are currently using account of "{current_account.owner}" ({current_account.base_currency}).')

        while True:
            print('\n' + '-'*40)
            print('           ACCOUNT MENU')
            print('-'*40)
            print('\nWhat do you want to do?')
            print('1) Check balance')
            print('2) Deposit money')
            print('3) Withdraw money')
            print('4) Transfer money')
            print('5) Go back to account selection')
            action_choice = input('\nChoose 1-5: ').strip()

            if action_choice == '1':
                current_account.show_balance()
            elif action_choice == '2':
                try:
                    currency = get_valid_currency('Enter deposit currency: ')
                    amount = float(input('Enter deposit amount: ').strip())
                    current_account.deposit(amount, currency)
                except ValueError as e:
                    print(f'\n Error: {e}❌')
                except Exception as e:
                    print(f'\n Unexpected error: {e}❌')
            elif action_choice == '3':
                try:
                    currency = get_valid_currency('Enter withdrawal currency: ')
                    amount = float(input('Enter withdrawal amount: ').strip())
                    current_account.withdraw(amount, currency)
                except ValueError as e:
                    print(f'\n Error: {e}❌')
                except Exception as e:
                    print(f'\n Unexpected error: {e}❌')
            elif action_choice == '4':
                try:
                    if current_account == account1:
                        recipient = account2
                        recipient_name = account2.owner
                    else:
                        recipient = account1
                        recipient_name = account1.owner

                    print(f'\n Transfer to: {recipient_name} ({recipient.base_currency})📤')
                    currency = get_valid_currency('Enter transfer currency: ')
                    amount = float(input('Enter transfer amount: ').strip())
                    current_account.transfer_to(recipient, amount, currency)
                except ValueError as e:
                    print(f'\n Error: {e}❌')
                except Exception as e:
                    print(f'\n Unexpected error: {e}❌')
            elif action_choice == '5':
                print(f'\nReturning to main menu...🔙 ')
                break
            else:
                print('\nInvalid choice. Try again.❌ ')

if __name__ == "__main__":
    main()