"""Find the deployed contract address from the deployer's transaction history."""
import os
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv("SEPOLIA_RPC_URL")))
pk = os.getenv("PRIVATE_KEY", "")
if not pk.startswith("0x"):
    pk = "0x" + pk

acct = w3.eth.account.from_key(pk)
nonce = w3.eth.get_transaction_count(acct.address)
balance = w3.from_wei(w3.eth.get_balance(acct.address), "ether")

print(f"Account: {acct.address}")
print(f"Nonce:   {nonce}")
print(f"Balance: {balance} ETH")

# Compute contract addresses for recent nonces
# Contract address = keccak256(rlp.encode([sender, nonce]))[12:]
for n in range(nonce):
    # Use web3 to compute CREATE address
    import rlp
    addr_bytes = w3.keccak(rlp.encode([bytes.fromhex(acct.address[2:]), n]))[12:]
    contract_addr = Web3.to_checksum_address(addr_bytes.hex())
    code = w3.eth.get_code(contract_addr)
    if len(code) > 0:
        print(f"\nContract found at nonce {n}:")
        print(f"  CONTRACT_ADDRESS={contract_addr}")
