"""
fix_nonce.py — Diagnose and flush stuck pending transactions on Sepolia.
Run this ONCE if enrollment TXs are not confirming.
"""
import os, sys
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY     = os.getenv("PRIVATE_KEY", "")
if not PRIVATE_KEY.startswith("0x"):
    PRIVATE_KEY = "0x" + PRIVATE_KEY

w3      = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

confirmed_nonce = w3.eth.get_transaction_count(account.address, "latest")
pending_nonce   = w3.eth.get_transaction_count(account.address, "pending")
balance         = w3.from_wei(w3.eth.get_balance(account.address), "ether")

print(f"Account : {account.address}")
print(f"Balance : {balance:.6f} ETH")
print(f"Confirmed nonce : {confirmed_nonce}")
print(f"Pending nonce   : {pending_nonce}")

stuck = pending_nonce - confirmed_nonce
if stuck == 0:
    print("\n✓ No stuck transactions. Nonce is clean.")
    sys.exit(0)

print(f"\n[!] {stuck} stuck/pending TX(s) detected.")
print("Sending replacement 0-ETH self-transfer to flush each stuck nonce...\n")

gas_price = int(w3.eth.gas_price * 2)  # 2x current gas price to replace stuck TXs

for nonce in range(confirmed_nonce, pending_nonce):
    tx = {
        "from":     account.address,
        "to":       account.address,
        "value":    0,
        "gas":      21000,
        "gasPrice": gas_price,
        "nonce":    nonce,
        "chainId":  11155111,  # Sepolia
    }
    signed   = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  Replacement TX for nonce {nonce}: {tx_hash.hex()}")
    print(f"  Waiting for confirmation...", end="", flush=True)
    receipt  = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=240)
    print(f"  ✓ Block #{receipt.blockNumber}")

print("\n✓ All stuck nonces flushed. You can now run interactive_demo.py normally.")
