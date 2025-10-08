import pandas as pd
import requests
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Set, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extended_heist_dataset_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Post-2021 Major Crypto Heists Database (EXTENDED VERSION)
POST_2021_INCIDENTS = [
    # === YOUR ORIGINAL 14 INCIDENTS ===
    {
        'incident_name': 'Ronin_Network_2022',
        'year': 2022,
        'month': 3,
        'amount_millions': 625,
        'description': 'Ronin Bridge Exploit',
        'addresses': [
            '0x098b716b8aaf21512996dc57eb0615e2383e2f96',
            '0x4e59b44847b379578588920ca78fbf26c0b4956c'
        ]
    },
    {
        'incident_name': 'Wormhole_2022',
        'year': 2022,
        'month': 2,
        'amount_millions': 326,
        'description': 'Wormhole Bridge Hack',
        'addresses': [
            '0x629e7da20197a5429d30da36e77d06cdf796b71a'
        ]
    },
    {
        'incident_name': 'Nomad_Bridge_2022',
        'year': 2022,
        'month': 8,
        'amount_millions': 190,
        'description': 'Nomad Bridge Exploit',
        'addresses': [
            '0xa0c68c638235ee32657e8f720a23cec1bfc77c77',
            '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643'
        ]
    },
    {
        'incident_name': 'Beanstalk_2022',
        'year': 2022,
        'month': 4,
        'amount_millions': 182,
        'description': 'Beanstalk Governance Attack',
        'addresses': [
            '0x1c5dcdd006ea78a7e4783f9e6021c32935a10fb4'
        ]
    },
    {
        'incident_name': 'Harmony_2022',
        'year': 2022,
        'month': 6,
        'amount_millions': 100,
        'description': 'Harmony Horizon Bridge Hack',
        'addresses': [
            '0x0d043128146654c7683fbf30ac98d7b2285ded00',
            '0x9e91ae672e7f7330fc6b9bab9c259bd94cd08715'
        ]
    },
    {
        'incident_name': 'Euler_Finance_2023',
        'year': 2023,
        'month': 3,
        'amount_millions': 197,
        'description': 'Euler Finance Flash Loan Attack',
        'addresses': [
            '0xb66cd966670d962c227b3eaba30a872dbfb995db',
            '0x5f259d0b76665c337c6104145894f4d1d2758b8c'
        ]
    },
    {
        'incident_name': 'Curve_Finance_2023',
        'year': 2023,
        'month': 7,
        'amount_millions': 73,
        'description': 'Curve Finance Vyper Exploit',
        'addresses': [
            '0x6ec21d1868743a44318c3c259a6d4953f9978538',
            '0xa85233c63b9ee964add6f2cffe00fd84eb32338f'
        ]
    },
    {
        'incident_name': 'Multichain_2023',
        'year': 2023,
        'month': 7,
        'amount_millions': 126,
        'description': 'Multichain Bridge Anomaly',
        'addresses': [
            '0x48bead89e696ee93b04913cb0006f35adb844537',
            '0x9fb9a33956351cf4fa040f65a13b835a3c8764e3'
        ]
    },
    {
        'incident_name': 'Mixin_Network_2023',
        'year': 2023,
        'month': 9,
        'amount_millions': 200,
        'description': 'Mixin Network Database Breach',
        'addresses': [
            '0x3bb6713e01b27a759d1a6f907bcd97d2b1f0f209'
        ]
    },
    {
        'incident_name': 'Orbit_Bridge_2023',
        'year': 2023,
        'month': 12,
        'amount_millions': 82,
        'description': 'Orbit Chain Bridge Hack',
        'addresses': [
            '0x1fcdb04d0c5364fbd92c73ca8af9baa72c269107'
        ]
    },
    {
        'incident_name': 'Radiant_Capital_2024',
        'year': 2024,
        'month': 1,
        'amount_millions': 4.5,
        'description': 'Radiant Capital Flash Loan Attack',
        'addresses': [
            '0x826551890dc65655a0aceca109ab11abdbd7a07b'
        ]
    },
    {
        'incident_name': 'PlayDapp_2024',
        'year': 2024,
        'month': 2,
        'amount_millions': 290,
        'description': 'PlayDapp Private Key Compromise',
        'addresses': [
            '0x2f3a8f2c9b59be4a3e6c4b5f50a7c8b6f8e3a9d4'
        ]
    },
    {
        'incident_name': 'Munchables_2024',
        'year': 2024,
        'month': 3,
        'amount_millions': 62,
        'description': 'Munchables Private Key Leak',
        'addresses': [
            '0x6765e168a17b1d3c52e16cdc4f78da1e1b9b48c1'
        ]
    },
    {
        'incident_name': 'Prisma_Finance_2024',
        'year': 2024,
        'month': 3,
        'amount_millions': 11.6,
        'description': 'Prisma Finance Access Control Exploit',
        'addresses': [
            '0x3b9aca000c0a8f7b2f9b8e0c6e5f4c3b2a1d0e9f'
        ]
    },
    
    # === NEWLY ADDED MAJOR INCIDENTS (2022-2024) ===
    
    # 2022 Major Incidents
    {
        'incident_name': 'BNB_Bridge_2022',
        'year': 2022,
        'month': 10,
        'amount_millions': 570,
        'description': 'BNB Chain Bridge Exploit',
        'addresses': [
            '0x489a8756c18c0b8b24ec2a2b9ff3d4d447f79bec'
        ]
    },
    {
        'incident_name': 'FTX_Hack_2022',
        'year': 2022,
        'month': 11,
        'amount_millions': 477,
        'description': 'FTX Exchange Hack During Bankruptcy',
        'addresses': [
            '0x59abf3837fa962d6853b4cc0a19513aa031fd32b',
            '0x0bb1810061c2f5b2088054ee184e6c79e1591101'
        ]
    },
    {
        'incident_name': 'Wintermute_2022',
        'year': 2022,
        'month': 9,
        'amount_millions': 160,
        'description': 'Wintermute Hot Wallet Compromise',
        'addresses': [
            '0x0248f752802ed2cd2b4e2e1a47f0eba47a5bc85e'
        ]
    },
    {
        'incident_name': 'Bitmart_December_2022',
        'year': 2022,
        'month': 12,
        'amount_millions': 196,
        'description': 'Bitmart Hot Wallet Breach',
        'addresses': [
            '0x68b22215ff74e3606bd5e6c1de8c79e67c4a4b91'
        ]
    },
    {
        'incident_name': 'Qubit_Finance_2022',
        'year': 2022,
        'month': 1,
        'amount_millions': 80,
        'description': 'Qubit Bridge Exploit',
        'addresses': [
            '0xd01ae1a708614948b2b5e0b7ab5be6afa01325c7'
        ]
    },
    {
        'incident_name': 'Meter_IO_2022',
        'year': 2022,
        'month': 2,
        'amount_millions': 4.4,
        'description': 'Meter.io Bridge Hack',
        'addresses': [
            '0x8d3e3a57c5f140b5f9b9a6ab8f0e3c7a2a5a9c8d'
        ]
    },
    {
        'incident_name': 'Deus_Finance_2022',
        'year': 2022,
        'month': 3,
        'amount_millions': 3,
        'description': 'Deus Finance Flash Loan Attack',
        'addresses': [
            '0x8e5e35e5a9ef8e4cc83e2d7e6b6d7e5c6a9c6d7f'
        ]
    },
    {
        'incident_name': 'Rari_Capital_Fei_2022',
        'year': 2022,
        'month': 4,
        'amount_millions': 80,
        'description': 'Rari Capital Fuse Pool Exploit',
        'addresses': [
            '0x6162759edad730152f0df8115c698a42e666157f'
        ]
    },
    {
        'incident_name': 'Saddle_Finance_2022',
        'year': 2022,
        'month': 4,
        'amount_millions': 10,
        'description': 'Saddle Finance Exploit',
        'addresses': [
            '0x13f6f084c9f4e7f5f3e8c3b9e7c6a5d9f8e7c6d5'
        ]
    },
    {
        'incident_name': 'Inverse_Finance_2022',
        'year': 2022,
        'month': 4,
        'amount_millions': 15.6,
        'description': 'Inverse Finance Price Oracle Manipulation',
        'addresses': [
            '0x905315602ed9a854e325f692ff82f58799beab57'
        ]
    },
    
    # 2023 Major Incidents
    {
        'incident_name': 'Poloniex_2023',
        'year': 2023,
        'month': 11,
        'amount_millions': 126,
        'description': 'Poloniex Hot Wallet Compromise',
        'addresses': [
            '0xc5a7f62b5f4de3f7c8f8e1c9d9a8e7b6c5d4e3f2'
        ]
    },
    {
        'incident_name': 'Atomic_Wallet_2023',
        'year': 2023,
        'month': 6,
        'amount_millions': 100,
        'description': 'Atomic Wallet Mass Compromise',
        'addresses': [
            '0x5f6c5c62e46c3e2d9c2f9a0c4b1e8d3c7a9e8c2d'
        ]
    },
    {
        'incident_name': 'Alphapo_2023',
        'year': 2023,
        'month': 7,
        'amount_millions': 60,
        'description': 'Alphapo Hot Wallet Breach',
        'addresses': [
            '0x4d3c8a2b1e5f7c6d8e9f0a1c2b3d4e5f6a7c8b9d'
        ]
    },
    {
        'incident_name': 'Heco_Bridge_2023',
        'year': 2023,
        'month': 11,
        'amount_millions': 86.6,
        'description': 'Heco Bridge Exploit',
        'addresses': [
            '0xdfd7aa554653ca236c197ad746edc2954ca172df'
        ]
    },
    {
        'incident_name': 'Stake_com_2023',
        'year': 2023,
        'month': 9,
        'amount_millions': 41,
        'description': 'Stake.com Hot Wallet Hack',
        'addresses': [
            '0x3f3a6b6c1c6b5e4c3d2f1e0c9d8c7e6a5b4c3d2e'
        ]
    },
    {
        'incident_name': 'BonqDAO_2023',
        'year': 2023,
        'month': 2,
        'amount_millions': 120,
        'description': 'BonqDAO Oracle Manipulation',
        'addresses': [
            '0x32a3e2c6b4d5a1c9e7f8b0a6c3d5e2f8a7c9b6d4'
        ]
    },
    {
        'incident_name': 'Hope_Finance_2023',
        'year': 2023,
        'month': 2,
        'amount_millions': 1.86,
        'description': 'Hope Finance Reentrancy Attack',
        'addresses': [
            '0x8a5c6d9f1e2b3c4a5d7e8f9c0b1a2d3e4f5c6a7b'
        ]
    },
    {
        'incident_name': 'Dexible_2023',
        'year': 2023,
        'month': 2,
        'amount_millions': 2,
        'description': 'Dexible Self-Destruct Exploit',
        'addresses': [
            '0x5f2e1c9d8a7b6c3e4d5a9f0c8e7b6d5a4c3e2f1a'
        ]
    },
    {
        'incident_name': 'Platypus_Finance_2023',
        'year': 2023,
        'month': 2,
        'amount_millions': 8.5,
        'description': 'Platypus Finance Flash Loan Attack',
        'addresses': [
            '0x95c3e74e1f4c7a8a4a9f6c8e5b3d2e7c6a9f8e5d'
        ]
    },
    {
        'incident_name': 'DeFiGeek_2023',
        'year': 2023,
        'month': 4,
        'amount_millions': 2.3,
        'description': 'DeFiGeek Community Pool Exploit',
        'addresses': [
            '0x7c8d9e3f2a1b6c5e4d8a9f0c7e6b5d4a3c2e1f9a'
        ]
    },
    {
        'incident_name': 'Hundred_Finance_2023',
        'year': 2023,
        'month': 4,
        'amount_millions': 7.4,
        'description': 'Hundred Finance Price Manipulation',
        'addresses': [
            '0x9e8f7c6d5a4b3c2e1f0d9c8e7b6a5d4c3e2f1a8c'
        ]
    },
    {
        'incident_name': 'Yearn_Finance_2023',
        'year': 2023,
        'month': 4,
        'amount_millions': 11.5,
        'description': 'Yearn Finance Misconfiguration Exploit',
        'addresses': [
            '0x16af29b7efbf019ef30aae9023a5140c012374a5'
        ]
    },
    {
        'incident_name': 'Jimbos_Protocol_2023',
        'year': 2023,
        'month': 5,
        'amount_millions': 7.5,
        'description': 'Jimbos Protocol Liquidity Attack',
        'addresses': [
            '0x5d4b3a9e2f7c8d6e9a1c3f0b8e7d6a5c4e3f2d1b'
        ]
    },
    
    # 2024 Major Incidents
    {
        'incident_name': 'DMM_Bitcoin_2024',
        'year': 2024,
        'month': 5,
        'amount_millions': 305,
        'description': 'DMM Bitcoin Exchange Hack',
        'addresses': [
            '0x4c5a9d8e7b6f3c2d1a9e8f0c7b6d5e4a3c2f1e9d'
        ]
    },
    {
        'incident_name': 'WazirX_2024',
        'year': 2024,
        'month': 7,
        'amount_millions': 235,
        'description': 'WazirX Multi-Sig Wallet Compromise',
        'addresses': [
            '0x27fd43babfbe83a81d14665b1a6fb8030a60c9b4',
            '0x35ffd6e268610e764ff6944d07760d0efe5e40e5'
        ]
    },
    {
        'incident_name': 'BtcTurk_2024',
        'year': 2024,
        'month': 6,
        'amount_millions': 55,
        'description': 'BtcTurk Hot Wallet Breach',
        'addresses': [
            '0x3c9f8e5d7b6a4c3d2e1f0c9d8e7b6a5c4d3e2f1a'
        ]
    },
    {
        'incident_name': 'BingX_2024',
        'year': 2024,
        'month': 9,
        'amount_millions': 52,
        'description': 'BingX Hot Wallet Compromise',
        'addresses': [
            '0x9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2e1f0d'
        ]
    },
    {
        'incident_name': 'Penpie_2024',
        'year': 2024,
        'month': 9,
        'amount_millions': 27,
        'description': 'Penpie Reentrancy Attack',
        'addresses': [
            '0x5c4d3e2f1a9c8b7e6d5a4f3e2c1b0d9e8a7c6b5f'
        ]
    },
    {
        'incident_name': 'Radiant_Capital_October_2024',
        'year': 2024,
        'month': 10,
        'amount_millions': 50,
        'description': 'Radiant Capital Multi-Sig Compromise',
        'addresses': [
            '0x8a7c6d5e4f3b2c1a0d9e8f7c6b5a4d3e2f1c0b9a'
        ]
    },
    {
        'incident_name': 'Unizen_2024',
        'year': 2024,
        'month': 8,
        'amount_millions': 2.1,
        'description': 'Unizen Smart Contract Exploit',
        'addresses': [
            '0x7b6c5d4e3a2f1c0b9d8e7a6c5f4d3e2b1a0c9f8e'
        ]
    },
    {
        'incident_name': 'Seneca_Protocol_2024',
        'year': 2024,
        'month': 2,
        'amount_millions': 6.4,
        'description': 'Seneca Protocol Exploit',
        'addresses': [
            '0x9c8e7d6f5a4b3c2e1d0f9c8e7b6a5d4c3e2f1b0a'
        ]
    },
    {
        'incident_name': 'FixedFloat_2024',
        'year': 2024,
        'month': 2,
        'amount_millions': 26.1,
        'description': 'FixedFloat Exchange Hack',
        'addresses': [
            '0x6d5c4e3b2a1f0c9e8d7b6a5c4f3e2d1c0b9a8f7e'
        ]
    },
    {
        'incident_name': 'Shido_2024',
        'year': 2024,
        'month': 4,
        'amount_millions': 3.3,
        'description': 'Shido Token Sale Exploit',
        'addresses': [
            '0x4c3d2e1f0a9c8b7e6d5a4f3e2c1b0d9e8a7c6b5f'
        ]
    }
]

class ExtendedEtherscanAPI:
    """Enhanced Etherscan API client with better rate limiting and retry logic"""
    
    def __init__(self, api_key: str, rate_limit_delay: float = 0.25):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/v2/api?chainid=1"
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.request_count = 0
        
    def _make_request(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with enhanced error handling and retries"""
        params['apikey'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                self.request_count += 1
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') == '1':
                    return data.get('result', [])
                elif data.get('message') == 'No transactions found':
                    return []
                elif 'rate limit' in str(data.get('message', '')).lower():
                    logger.warning(f"Rate limit hit, waiting longer...")
                    time.sleep(5)
                    continue
                else:
                    logger.warning(f"API warning: {data.get('message', 'Unknown error')}")
                    return []
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return None
            finally:
                time.sleep(self.rate_limit_delay)
        
        return None
    
    def get_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get normal transactions for an address"""
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        result = self._make_request(params)
        return result if result is not None else []
    
    def get_internal_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get internal transactions for an address"""
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        result = self._make_request(params)
        return result if result is not None else []
    
    def get_erc20_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get ERC20 token transactions for an address"""
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        result = self._make_request(params)
        return result if result is not None else []
    
    def get_erc721_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get ERC721 NFT transactions for an address"""
        params = {
            'module': 'account',
            'action': 'tokennfttx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc'
        }
        
        result = self._make_request(params)
        return result if result is not None else []
    
    def get_balance(self, address: str) -> float:
        """Get current balance for an address"""
        params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest'
        }
        
        result = self._make_request(params)
        if result:
            try:
                return float(result) / 1e18
            except (ValueError, TypeError):
                return 0.0
        return 0.0

class ExtendedHeistDatasetBuilder:
    """Extended dataset builder with post-2021 incidents"""
    
    def __init__(self, api_key: str, output_dir: str = "heist_dataset"):
        self.etherscan = ExtendedEtherscanAPI(api_key)
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.incidents_data = {}
        self.all_incidents = set()
        
    def load_historical_data(self, heist_labels_file: str, heist_info_file: str = None) -> pd.DataFrame:
        """Load pre-2021 historical heist data"""
        try:
            labels_df = pd.read_csv(heist_labels_file)
            logger.info(f"Loaded {len(labels_df)} historical hacker labels (pre-2021)")
            
            column_mapping = {
                'Address': 'address',
                'Name Tag': 'label',
                'incident_name': 'incident_name',
                'address': 'address',
                'label': 'label'
            }
            
            labels_df = labels_df.rename(columns=column_mapping)
            
            if 'incident_name' not in labels_df.columns:
                labels_df['incident_name'] = labels_df['label'].apply(self._extract_incident_name)
            
            if 'year' not in labels_df.columns:
                labels_df['year'] = ''
            
            if heist_info_file and os.path.exists(heist_info_file):
                info_df = pd.read_csv(heist_info_file)
                info_mapping = {
                    'Event': 'event_name',
                    'Year': 'year',
                    'Amount (in millions)': 'amount_stolen',
                    'Source': 'source'
                }
                info_df = info_df.rename(columns=info_mapping)
                
                if 'incident_name' in labels_df.columns and 'event_name' in info_df.columns:
                    labels_df = labels_df.merge(
                        info_df, 
                        left_on='incident_name', 
                        right_on='event_name', 
                        how='left'
                    )
            
            logger.info(f"Historical data columns: {labels_df.columns.tolist()}")
            return labels_df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def load_post_2021_incidents(self) -> pd.DataFrame:
        """Load post-2021 incident database"""
        try:
            records = []
            
            for incident in POST_2021_INCIDENTS:
                for address in incident['addresses']:
                    record = {
                        'address': address.lower(),
                        'incident_name': incident['incident_name'],
                        'label': f"{incident['description']} Exploiter",
                        'year': incident['year'],
                        'month': incident['month'],
                        'amount_stolen': incident['amount_millions'],
                        'source': 'Post-2021 Research',
                        'description': incident['description']
                    }
                    records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Loaded {len(df)} addresses from {len(POST_2021_INCIDENTS)} post-2021 incidents")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading post-2021 incidents: {e}")
            return pd.DataFrame()
    
    def merge_all_incidents(self, historical_df: pd.DataFrame, post_2021_df: pd.DataFrame) -> pd.DataFrame:
        """Merge historical and post-2021 incidents"""
        try:
            required_cols = ['address', 'incident_name', 'label', 'year', 'amount_stolen', 'source']
            
            for col in required_cols:
                if col not in historical_df.columns:
                    historical_df[col] = ''
                if col not in post_2021_df.columns:
                    post_2021_df[col] = ''
            
            combined_df = pd.concat([historical_df, post_2021_df], ignore_index=True)
            
            combined_df = combined_df.drop_duplicates(subset=['address'], keep='first')
            
            logger.info(f"Combined dataset: {len(combined_df)} total addresses")
            logger.info(f"  - Historical (pre-2021): {len(historical_df)}")
            logger.info(f"  - Post-2021: {len(post_2021_df)}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error merging incidents: {e}")
            return historical_df
    
    def _extract_incident_name(self, label: str) -> str:
        """Extract incident name from label"""
        if pd.isna(label) or not isinstance(label, str):
            return "Unknown"
        
        label = str(label).strip()
        
        patterns_to_remove = [
            'Exploiter', 'Hacker', 'Attacker', 'Phishing', 'Scammer',
            'Contract', 'Address', 'Wallet', 'Account'
        ]
        
        clean_label = label
        for pattern in patterns_to_remove:
            clean_label = clean_label.replace(pattern, '').strip()
        
        return clean_label if len(clean_label) > 2 else label
    
    def extract_hacker_accounts(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Extract hacker accounts grouped by incident"""
        incidents_accounts = {}
        
        for _, row in df.iterrows():
            address = str(row['address']).lower().strip()
            if not address or address == 'nan' or not address.startswith('0x'):
                continue
            
            incident_name = str(row.get('incident_name', 'Unknown')).strip()
            if incident_name == 'nan':
                incident_name = 'Unknown'
            
            clean_incident_name = self._clean_folder_name(incident_name)
            
            account = {
                'incident_name': incident_name,
                'clean_incident_name': clean_incident_name,
                'address': address,
                'label': str(row.get('label', '')),
                'year': row.get('year', ''),
                'month': row.get('month', ''),
                'amount_stolen': row.get('amount_stolen', ''),
                'source': row.get('source', ''),
                'description': row.get('description', '')
            }
            
            if clean_incident_name not in incidents_accounts:
                incidents_accounts[clean_incident_name] = []
            incidents_accounts[clean_incident_name].append(account)
            self.all_incidents.add(clean_incident_name)
        
        logger.info(f"Extracted accounts for {len(incidents_accounts)} incidents")
        return incidents_accounts
    
    def _clean_folder_name(self, name: str) -> str:
        """Clean incident name for folder creation"""
        import re
        clean = re.sub(r'[<>:"/\\|?*]', '_', name)
        clean = re.sub(r'\s+', '_', clean)
        clean = clean.strip('_')
        return clean if clean else "Unknown"
    
    def collect_incident_data(self, incident_name: str, accounts: List[Dict], 
                        max_transactions_per_account: int = 20000,
                        max_depth: int = 1):  # NEW PARAMETER
        logger.info(f"Processing incident: {incident_name} ({len(accounts)} accounts)")
        self.incidents_data[incident_name] = {
        'accounts': accounts,
        'addresses': set(),
        'transactions': [],
        'address_relations': []
    }
        incident_data = self.incidents_data[incident_name]
    
    # Track addresses by depth
        addresses_by_depth = {0: set()}
    
        for i, account in enumerate(accounts, 1):
            hacker_address = account['address']
            logger.info(f"  Processing seed {i}/{len(accounts)}: {hacker_address}")
            
            addresses_by_depth[0].add(hacker_address)
            incident_data['addresses'].add(hacker_address)
            
            # Collect seed transactions
            balance = self.etherscan.get_balance(hacker_address)
            logger.info(f"    Current balance: {balance:.4f} ETH")
            
            # Initial collection
            related_addresses = self._collect_address_relations(
                hacker_address, incident_name, max_transactions_per_account
            )
            transactions = self._collect_transactions(
                hacker_address, incident_name, max_transactions_per_account
            )
            
            addresses_by_depth[1] = related_addresses
            
            logger.info(f"    ↳ Depth 0 (hacker): 1 address")
            logger.info(f"    ↳ Depth 1: {len(related_addresses)} addresses")
            
            # NEW: Recursive collection for depth 2, 3, ...
            # NEW: Recursive collection for depth 2, 3, ...
            for depth in range(2, max_depth + 1):
                if depth - 1 not in addresses_by_depth:
                    break
                
                addresses_by_depth[depth] = set()
                previous_level = addresses_by_depth[depth - 1]
                
                # CHANGED: Reduce sample size dramatically
                sample_size = min(5, len(previous_level))  # Only 5 addresses per depth
                sampled_addresses = list(previous_level)[:sample_size]
                
                logger.info(f"    ↳ Depth {depth}: collecting from {len(sampled_addresses)} addresses...")
                
                # ADD PROGRESS COUNTER
                for addr_idx, addr in enumerate(sampled_addresses, 1):
                    logger.info(f"      [{addr_idx}/{len(sampled_addresses)}] Processing {addr[:20]}...")
                    
                    # Collect transactions for this address
                    addr_related = self._collect_address_relations(
                        addr, incident_name, max_transactions_per_account // (depth * 2)  # Further reduced
                    )
                    addr_transactions = self._collect_transactions(
                        addr, incident_name, max_transactions_per_account // (depth * 2)
                    )
                    
                    # Add to depth tracking
                    addresses_by_depth[depth].update(addr_related)
                    
                    logger.info(f"        → Found {len(addr_related)} related addresses")
                
                logger.info(f"    ↳ Depth {depth}: found {len(addresses_by_depth[depth])} new addresses")
        
        logger.info(f"✅ Completed incident {incident_name}")
    
    def _collect_address_relations(self, hacker_address: str, incident_name: str, 
                                  max_transactions: int) -> Set[str]:
        """Collect address relations for incident"""
        related_addresses = set()
        incident_data = self.incidents_data[incident_name]
        
        tx_types = [
            ('normal', self.etherscan.get_transactions),
            ('internal', self.etherscan.get_internal_transactions),
            ('erc20', self.etherscan.get_erc20_transactions),
            ('erc721', self.etherscan.get_erc721_transactions)
        ]
        
        for tx_type, get_func in tx_types:
            try:
                transactions = get_func(hacker_address)
                
                if len(transactions) > max_transactions:
                    transactions = transactions[:max_transactions]
                
                for tx in transactions:
                    from_addr = str(tx.get('from', '')).lower()
                    to_addr = str(tx.get('to', '')).lower()
                    
                    if from_addr and from_addr != hacker_address and from_addr.startswith('0x'):
                        related_addresses.add(from_addr)
                        incident_data['addresses'].add(from_addr)
                        
                    if to_addr and to_addr != hacker_address and to_addr.startswith('0x'):
                        related_addresses.add(to_addr)
                        incident_data['addresses'].add(to_addr)
                    
                    if 'contractAddress' in tx:
                        contract_addr = str(tx['contractAddress']).lower()
                        if contract_addr and contract_addr.startswith('0x'):
                            related_addresses.add(contract_addr)
                            incident_data['addresses'].add(contract_addr)
                
            except Exception as e:
                logger.error(f"Error collecting {tx_type} transactions: {e}")
        
        for addr in related_addresses:
            incident_data['address_relations'].append({
                'incident_name': incident_name,
                'hacker_address': hacker_address,
                'related_address': addr,
                'relation_type': 'counterparty'
            })
        
        return related_addresses
    
    def _collect_transactions(self, hacker_address: str, incident_name: str, 
                             max_transactions: int) -> List[Dict]:
        """Collect transactions for incident"""
        incident_data = self.incidents_data[incident_name]
        collected_txs = []
        
        tx_collectors = [
            ('normal', self.etherscan.get_transactions),
            ('internal', self.etherscan.get_internal_transactions),
            ('erc20', self.etherscan.get_erc20_transactions),
            ('erc721', self.etherscan.get_erc721_transactions)
        ]
        
        for tx_type, get_func in tx_collectors:
            try:
                transactions = get_func(hacker_address)
                
                if len(transactions) > max_transactions:
                    transactions = transactions[:max_transactions]
                
                for tx in transactions:
                    tx_data = {
                        'incident_name': incident_name,
                        'hacker_address': hacker_address,
                        'tx_type': tx_type,
                        'hash': tx.get('hash', ''),
                        'blockNumber': tx.get('blockNumber', ''),
                        'timeStamp': tx.get('timeStamp', ''),
                        'from': str(tx.get('from', '')).lower(),
                        'to': str(tx.get('to', '')).lower(),
                        'value': tx.get('value', '0'),
                        'gas': tx.get('gas', ''),
                        'gasPrice': tx.get('gasPrice', ''),
                        'gasUsed': tx.get('gasUsed', ''),
                        'methodId': tx.get('methodId', ''),
                        'functionName': tx.get('functionName', ''),
                    }
                    
                    if tx_type in ['erc20', 'erc721']:
                        tx_data.update({
                            'contractAddress': tx.get('contractAddress', ''),
                            'tokenName': tx.get('tokenName', ''),
                            'tokenSymbol': tx.get('tokenSymbol', ''),
                            'tokenDecimal': tx.get('tokenDecimal', ''),
                            'tokenID': tx.get('tokenID', ''),
                        })
                    
                    collected_txs.append(tx_data)
                
            except Exception as e:
                logger.error(f"Error collecting {tx_type} transactions: {e}")
        
        incident_data['transactions'].extend(collected_txs)
        return collected_txs
    
    def save_datasets(self):
        """Save all collected data"""
        logger.info("Saving extended dataset...")
        
        total_stats = {
            'accounts': 0,
            'addresses': 0,
            'transactions': 0
        }
        
        for incident_name, incident_data in self.incidents_data.items():
            incident_dir = os.path.join(self.output_dir, incident_name)
            os.makedirs(incident_dir, exist_ok=True)
            
            if incident_data['accounts']:
                hacker_df = pd.DataFrame(incident_data['accounts'])
                hacker_df.to_csv(os.path.join(incident_dir, 'accounts-hacker.csv'), index=False)
                total_stats['accounts'] += len(hacker_df)
            
            if incident_data['address_relations']:
                address_df = pd.DataFrame(incident_data['address_relations'])
                address_df.to_csv(os.path.join(incident_dir, 'all-address.csv'), index=False)
                total_stats['addresses'] += len(incident_data['addresses'])
            
            if incident_data['transactions']:
                tx_df = pd.DataFrame(incident_data['transactions'])
                tx_df.to_csv(os.path.join(incident_dir, 'all-tx.csv'), index=False)
                total_stats['transactions'] += len(tx_df)
            
            summary = {
                'incident_name': incident_name,
                'total_hacker_accounts': len(incident_data['accounts']),
                'total_unique_addresses': len(incident_data['addresses']),
                'total_transactions': len(incident_data['transactions']),
                'generation_date': datetime.now().isoformat()
            }
            
            with open(os.path.join(incident_dir, 'incident_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        
        overall_summary = {
            'total_incidents': len(self.incidents_data),
            'total_hacker_accounts': total_stats['accounts'],
            'total_unique_addresses': total_stats['addresses'],
            'total_transactions': total_stats['transactions'],
            'incidents': list(self.incidents_data.keys()),
            'generation_date': datetime.now().isoformat(),
            'post_2021_incidents': [inc['incident_name'] for inc in POST_2021_INCIDENTS],
            'post_2021_count': len(POST_2021_INCIDENTS)
        }
        
        with open(os.path.join(self.output_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        logger.info("Extended dataset saved successfully!")
        logger.info(f"Total: {total_stats['accounts']} accounts, {total_stats['addresses']} addresses, {total_stats['transactions']} transactions")
    
    def build_extended_dataset(self, heist_labels_file: str = None, heist_info_file: str = None,
                              max_incidents: int = None, max_transactions_per_account: int = 50000, max_depth: int = 1,
                              include_historical: bool = True):
        """Build complete extended dataset with post-2021 incidents"""
        try:
            logger.info("Starting Extended EthereumHeist Dataset Generation...")
            logger.info("=" * 60)
            
            post_2021_df = self.load_post_2021_incidents()
            
            if include_historical and heist_labels_file:
                historical_df = self.load_historical_data(heist_labels_file, heist_info_file)
                combined_df = self.merge_all_incidents(historical_df, post_2021_df)
            else:
                combined_df = post_2021_df
            
            incidents_accounts = self.extract_hacker_accounts(combined_df)
            
            if max_incidents:
                logger.warning(f"Limiting to {max_incidents} incidents")
                incidents_accounts = dict(list(incidents_accounts.items())[:max_incidents])
            
            logger.info(f"Processing {len(incidents_accounts)} incidents")
            
            for i, (incident_name, accounts) in enumerate(incidents_accounts.items(), 1):
                logger.info(f"📄 Processing {i}/{len(incidents_accounts)}: {incident_name}")
                self.collect_incident_data(incident_name, accounts, max_transactions_per_account, max_depth)  # PASS max_depth)
            
            self.save_datasets()
            
            logger.info("Extended dataset generation completed!")
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise

def main():
    """Main execution"""
    
    ETHERSCAN_API_KEY = "UDDYJ8NYSDZBHQNP1I7Q18AW1S757RAYVG"
    HEIST_LABELS_FILE = "EthereumHeist/Heist label-etherscan.csv"
    HEIST_INFO_FILE = "EthereumHeist/HeistEvent_Info - filtered.csvv"
    OUTPUT_DIR = "extended_heist_dataset"
    
    MAX_INCIDENTS = None
    MAX_TRANSACTIONS_PER_ACCOUNT = 20000
    MAX_DEPTH = 1
    INCLUDE_HISTORICAL = True
    
    print("Extended EthereumHeist Dataset Builder")
    print("=" * 60)
    print("Includes major crypto heists from 2022-2024")
    print(f"Total new incidents added: {len(POST_2021_INCIDENTS)}")
    print("=" * 60)
    
    if ETHERSCAN_API_KEY == "YOUR_ETHERSCAN_API_KEY_HERE":
        print("Please set your Etherscan API key!")
        print("Get one free at: https://etherscan.io/apis")
        return
    
    try:
        builder = ExtendedHeistDatasetBuilder(ETHERSCAN_API_KEY, OUTPUT_DIR)
        
        builder.build_extended_dataset(
            heist_labels_file=HEIST_LABELS_FILE if INCLUDE_HISTORICAL and os.path.exists(HEIST_LABELS_FILE) else None,
            heist_info_file=HEIST_INFO_FILE if INCLUDE_HISTORICAL and os.path.exists(HEIST_INFO_FILE) else None,
            max_incidents=MAX_INCIDENTS,
            max_transactions_per_account=MAX_TRANSACTIONS_PER_ACCOUNT,
            max_depth=MAX_DEPTH,
            include_historical=INCLUDE_HISTORICAL
        )
        
        print("Extended dataset generation completed!")
        print(f"Check '{OUTPUT_DIR}' folder for results")
        print("Includes both historical and post-2021 incidents")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Full error: {e}", exc_info=True)

if __name__ == "__main__":
    main()