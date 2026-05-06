// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title STSMIRS — Smart Tourist Safety Monitoring and Incident Response System
 * @notice Privacy-preserving identity management for tourist safety.
 *         Zero personal data on-chain; only id_hash and ADM reference stored.
 */
contract STSMIRS {
    // ───────── State ─────────

    address public owner;

    mapping(address => bool) public authorisedServers;

    struct IdentityRecord {
        bytes32 idHash;        // SHA-256 of a random UUID (not PII-derived)
        string  admRef;        // UUID pointer into the off-chain KMD
        string  ownerPubKey;   // Tourist's public key (for future end-to-end consent)
        bool    exists;
    }

    mapping(bytes32 => IdentityRecord) public identities;

    struct AccessLog {
        string  emergencyType;
        uint256 score;
        uint256 timestamp;
        bool    released;
        address responder;
    }

    mapping(bytes32 => AccessLog[]) public accessLogs;

    // ───────── Events ─────────

    event IdentityEnrolled(
        bytes32 indexed idHash,
        string  admRef,
        uint256 timestamp
    );

    event EmergencyAccessGranted(
        bytes32 indexed idHash,
        string  emergencyType,
        uint256 score,
        uint256 timestamp
    );

    event ReleaseConfirmed(
        bytes32 indexed idHash,
        address responder,
        uint256 timestamp
    );

    // ───────── Modifiers ─────────

    modifier onlyOwner() {
        require(msg.sender == owner, "STSMIRS: caller is not the owner");
        _;
    }

    modifier onlyAuthorised() {
        require(
            authorisedServers[msg.sender],
            "STSMIRS: caller is not an authorised server"
        );
        _;
    }

    // ───────── Constructor ─────────

    constructor() {
        owner = msg.sender;
        authorisedServers[msg.sender] = true; // deployer is auto-authorised
    }

    // ───────── Admin ─────────

    function addAuthorisedServer(address server) external onlyOwner {
        authorisedServers[server] = true;
    }

    function removeAuthorisedServer(address server) external onlyOwner {
        authorisedServers[server] = false;
    }

    // ───────── Core Functions ─────────

    /**
     * @notice Enroll a tourist identity anchor on-chain.
     * @param idHash      SHA-256 hash of a random UUID (bytes32)
     * @param admRef      UUID string pointing to the KMD record
     * @param ownerPubKey Tourist's public key string
     */
    function enrollIdentity(
        bytes32 idHash,
        string calldata admRef,
        string calldata ownerPubKey
    ) external onlyAuthorised {
        require(!identities[idHash].exists, "STSMIRS: identity already enrolled");

        identities[idHash] = IdentityRecord({
            idHash:      idHash,
            admRef:      admRef,
            ownerPubKey: ownerPubKey,
            exists:      true
        });

        emit IdentityEnrolled(idHash, admRef, block.timestamp);
    }

    /**
     * @notice Request emergency access to a tourist's identity.
     * @param idHash        Identity hash of the tourist
     * @param emergencyType "SMALL_FIGHT", "VIOLENT_FIGHT", "MEDICAL_PROBLEM", "MEDICAL_EMERGENCY", or "OFFENCE"
     * @param score         Safety score (scaled by 100, e.g. 85 = 0.85)
     */
    function requestEmergencyAccess(
        bytes32 idHash,
        string calldata emergencyType,
        uint256 score
    ) external onlyAuthorised {
        require(identities[idHash].exists, "STSMIRS: identity not found");

        accessLogs[idHash].push(AccessLog({
            emergencyType: emergencyType,
            score:         score,
            timestamp:     block.timestamp,
            released:      false,
            responder:     address(0)
        }));

        emit EmergencyAccessGranted(idHash, emergencyType, score, block.timestamp);
    }

    /**
     * @notice Confirm that identity data was released to a responder (audit trail).
     * @param idHash    Identity hash of the tourist
     * @param responder Address of the authorised responder
     */
    function confirmRelease(
        bytes32 idHash,
        address responder
    ) external onlyAuthorised {
        require(identities[idHash].exists, "STSMIRS: identity not found");

        uint256 len = accessLogs[idHash].length;
        require(len > 0, "STSMIRS: no access request found");

        // Mark the latest access log as released
        accessLogs[idHash][len - 1].released  = true;
        accessLogs[idHash][len - 1].responder = responder;

        emit ReleaseConfirmed(idHash, responder, block.timestamp);
    }

    // ───────── View Helpers ─────────

    function getIdentity(bytes32 idHash)
        external view returns (IdentityRecord memory)
    {
        require(identities[idHash].exists, "STSMIRS: identity not found");
        return identities[idHash];
    }

    function getAccessLogCount(bytes32 idHash) external view returns (uint256) {
        return accessLogs[idHash].length;
    }

    function getAccessLog(bytes32 idHash, uint256 index)
        external view returns (AccessLog memory)
    {
        require(index < accessLogs[idHash].length, "STSMIRS: index out of bounds");
        return accessLogs[idHash][index];
    }
}
