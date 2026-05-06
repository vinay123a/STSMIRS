from web3 import Web3
import sys

# List of public Sepolia RPCs to try
RPC_NODES = [
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://rpc.sepolia.org",
    "https://gateway.tenderly.co/public/sepolia"
]

def check_transaction(tx_hash):
    # 1. Clean the hash
    tx_hash = tx_hash.strip()
    if not tx_hash.startswith('0x'):
        tx_hash = '0x' + tx_hash
    
    print(f"[*] Checking Hash: {tx_hash}")

    # 2. Try to connect to a working node
    w3 = None
    for url in RPC_NODES:
        temp_w3 = Web3(Web3.HTTPProvider(url))
        if temp_w3.is_connected():
            w3 = temp_w3
            print(f"[+] Connected successfully to: {url}")
            break
    
    if not w3:
        print("[-] ERROR: Could not connect to any Sepolia nodes. Check your internet.")
        return

    # 3. Fetch the data
    try:
        print("[*] Searching for transaction...")
        tx = w3.eth.get_transaction(tx_hash)
        
        print("\n=== SUCCESS: DATA FOUND ===")
        print(f"Sender (From): {tx['from']}")
        print(f"Receiver (To): {tx['to']}")
        print(f"Value:         {w3.from_wei(tx['value'], 'ether')} ETH")
        print(f"Block Number:  {tx['blockNumber']}")
        print(f"Nonce:         {tx['nonce']}")
        
        # This is the "Data" part of the transaction
        if tx['input'] != '0x':
            print(f"Input Data (Hex): {tx['input'].hex()}")
        else:
            print("Input Data: None (Standard ETH Transfer)")

    except Exception as e:
        print(f"[-] ERROR: Transaction not found on Ethereum Sepolia.")
        print(f"    Technical Message: {e}")
        print("\nTIP: Is this transaction on BASE Sepolia or ARBITRUM Sepolia instead?")

if __name__ == "__main__":
    # Your specific hash
    target_hash = "79f139f64b7f22fc3d4d57c435ab604ba701cfb7bafebca6bc5fd86bf31cedcf"
    check_transaction(target_hash)