import os, json
from web3 import Web3
from dotenv import load_dotenv
load_dotenv()
SEPOLIA_RPC_URL = os.getenv('SEPOLIA_RPC_URL')
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
if not SEPOLIA_RPC_URL or not CONTRACT_ADDRESS:
    raise SystemExit('MISSING_ENV')
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
abi = json.load(open('blockchain/artifacts/contracts/STSMIRS.sol/STSMIRS.json'))['abi']
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
print('connected', w3.is_connected())
print('block', w3.eth.block_number)
print('address', contract.address, type(contract.address))
print('event', contract.events.EmergencyAccessGranted)
print('has createFilter', hasattr(contract.events.EmergencyAccessGranted, 'createFilter'))
print('has create_filter', hasattr(contract.events.EmergencyAccessGranted, 'create_filter'))
print('has processLog', hasattr(contract.events.EmergencyAccessGranted(), 'processLog'))
print('has process_logs', hasattr(contract.events.EmergencyAccessGranted(), 'process_logs'))
print('dir event', [x for x in dir(contract.events.EmergencyAccessGranted) if 'create' in x or 'process' in x])
