const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying STSMIRS contract with account:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", ethers.formatEther(balance), "ETH");

  const STSMIRS = await ethers.getContractFactory("STSMIRS");
  const contract = await STSMIRS.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("STSMIRS deployed to:", address);
  console.log("\n=== SAVE THIS ADDRESS ===");
  console.log("Add to your Python .env file:");
  console.log(`CONTRACT_ADDRESS=${address}`);
  console.log("=========================\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
