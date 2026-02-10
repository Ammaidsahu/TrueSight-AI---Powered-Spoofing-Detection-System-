"""
Blockchain Evidence Logging Module
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import asyncio

class EvidenceLogger:
    """Blockchain-based evidence logging for immutable proof"""
    
    def __init__(self, blockchain_network: str = "fabric-testnet"):
        self.blockchain_network = blockchain_network
        self.contract_id = "evidence-contract"
        self.connected = False
        self._connect_to_blockchain()
    
    def _connect_to_blockchain(self):
        """Connect to blockchain network"""
        try:
            # In production, this would connect to Hyperledger Fabric or other blockchain
            # For simulation, we'll use a mock connection
            logger.info(f"Connecting to blockchain network: {self.blockchain_network}")
            
            # Simulate connection
            self.connected = True
            logger.info("Blockchain connection established")
            
        except Exception as e:
            logger.warning(f"Blockchain connection failed: {e}. Using simulation mode.")
            self.connected = False
    
    async def log_evidence(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log evidence to blockchain"""
        try:
            # Prepare evidence package
            evidence_package = await self._prepare_evidence_package(evidence_data)
            
            # Calculate evidence hash
            evidence_hash = self._calculate_evidence_hash(evidence_package)
            
            # Create blockchain transaction
            transaction = await self._create_transaction(evidence_hash, evidence_package)
            
            # Submit to blockchain
            tx_result = await self._submit_to_blockchain(transaction)
            
            # Verify confirmation
            confirmation = await self._verify_confirmation(tx_result)
            
            return {
                "evidence_id": evidence_package["evidence_id"],
                "transaction_hash": tx_result.get("transaction_hash"),
                "blockchain_timestamp": tx_result.get("timestamp"),
                "evidence_hash": evidence_hash,
                "status": "confirmed" if confirmation else "pending",
                "confirmation_details": confirmation,
                "logged_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evidence logging failed: {e}")
            raise
    
    async def _prepare_evidence_package(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare evidence data package for blockchain logging"""
        # Generate unique evidence ID
        evidence_id = self._generate_evidence_id(evidence_data)
        
        # Add timestamps and metadata
        package = {
            "evidence_id": evidence_id,
            "created_at": datetime.utcnow().isoformat(),
            "data_type": evidence_data.get("type", "unknown"),
            "source_system": "TrueSight v1.0",
            "version": "1.0",
            "data_payload": evidence_data,
            "chain_of_custody": [{
                "handler": "system",
                "action": "creation",
                "timestamp": datetime.utcnow().isoformat(),
                "location": "server"
            }]
        }
        
        return package
    
    def _generate_evidence_id(self, evidence_data: Dict[str, Any]) -> str:
        """Generate unique evidence identifier"""
        # Combine key evidence attributes
        seed_data = f"{evidence_data.get('media_file_id', '')}_{evidence_data.get('detection_id', '')}_{time.time()}"
        return f"EVID-{hashlib.sha256(seed_data.encode()).hexdigest()[:16].upper()}"
    
    def _calculate_evidence_hash(self, evidence_package: Dict[str, Any]) -> str:
        """Calculate cryptographic hash of evidence package"""
        # Serialize package for hashing
        serialized = json.dumps(evidence_package, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def _create_transaction(self, evidence_hash: str, evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        """Create blockchain transaction"""
        transaction = {
            "contract_id": self.contract_id,
            "function": "logEvidence",
            "arguments": {
                "evidenceHash": evidence_hash,
                "evidenceData": evidence_package,
                "timestamp": datetime.utcnow().isoformat()
            },
            "creator": "TrueSight-System",
            "nonce": self._generate_nonce()
        }
        
        return transaction
    
    def _generate_nonce(self) -> str:
        """Generate cryptographic nonce"""
        return hashlib.sha256(f"{time.time()}_{hashlib.sha256(str(time.time()).encode()).hexdigest()}".encode()).hexdigest()[:32]
    
    async def _submit_to_blockchain(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Submit transaction to blockchain"""
        if not self.connected:
            # Simulation mode
            return {
                "transaction_hash": f"0x{hashlib.sha256(json.dumps(transaction).encode()).hexdigest()}",
                "timestamp": datetime.utcnow().isoformat(),
                "block_number": 1234567,
                "gas_used": 21000,
                "status": "success"
            }
        
        # In production, this would submit to actual blockchain
        # await self.fabric_client.submit_transaction(transaction)
        
        return {
            "transaction_hash": "0x_simulated_hash",
            "timestamp": datetime.utcnow().isoformat(),
            "block_number": 1234567,
            "gas_used": 21000,
            "status": "success"
        }
    
    async def _verify_confirmation(self, tx_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify blockchain confirmation"""
        if not self.connected:
            # Simulation mode - immediate confirmation
            return {
                "confirmed": True,
                "confirmations": 1,
                "block_height": tx_result.get("block_number"),
                "verification_time_ms": 150
            }
        
        # In production, this would poll for confirmation
        # confirmation = await self.fabric_client.wait_for_confirmation(tx_result["transaction_hash"])
        
        return {
            "confirmed": True,
            "confirmations": 3,  # Wait for multiple confirmations in production
            "block_height": tx_result.get("block_number"),
            "verification_time_ms": 2500
        }
    
    async def verify_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """Verify evidence existence and integrity on blockchain"""
        try:
            # Query blockchain for evidence
            evidence_record = await self._query_evidence(evidence_id)
            
            if not evidence_record:
                return {
                    "evidence_id": evidence_id,
                    "status": "not_found",
                    "verified": False
                }
            
            # Verify evidence integrity
            integrity_check = await self._verify_evidence_integrity(evidence_record)
            
            return {
                "evidence_id": evidence_id,
                "status": "found",
                "verified": integrity_check["integrity_valid"],
                "blockchain_proof": evidence_record,
                "integrity_verification": integrity_check,
                "verification_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Evidence verification failed: {e}")
            return {
                "evidence_id": evidence_id,
                "status": "verification_error",
                "verified": False,
                "error": str(e)
            }
    
    async def _query_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Query blockchain for evidence record"""
        if not self.connected:
            # Simulation - return mock data
            return {
                "evidence_id": evidence_id,
                "transaction_hash": f"0x_mock_hash_{evidence_id}",
                "block_number": 1234567,
                "timestamp": "2026-02-10T10:30:00Z",
                "evidence_hash": "mock_evidence_hash",
                "data": {"type": "detection_result", "mock": True}
            }
        
        # In production: await self.fabric_client.query_contract(self.contract_id, "getEvidence", [evidence_id])
        return None
    
    async def _verify_evidence_integrity(self, evidence_record: Dict[str, Any]) -> Dict[str, Any]:
        """Verify evidence data integrity"""
        try:
            # Recalculate hash and compare
            recalculated_hash = hashlib.sha256(
                json.dumps(evidence_record["data"], sort_keys=True, default=str).encode()
            ).hexdigest()
            
            hash_match = recalculated_hash == evidence_record["evidence_hash"]
            
            return {
                "integrity_valid": hash_match,
                "expected_hash": evidence_record["evidence_hash"],
                "calculated_hash": recalculated_hash,
                "hash_match": hash_match,
                "timestamp_verified": self._verify_timestamp(evidence_record["timestamp"])
            }
            
        except Exception as e:
            return {
                "integrity_valid": False,
                "error": str(e),
                "hash_match": False
            }
    
    def _verify_timestamp(self, timestamp_str: str) -> bool:
        """Verify timestamp validity"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.utcnow()
            # Check if timestamp is reasonable (not in future, not too old)
            return timestamp <= now and (now - timestamp).days < 365
        except Exception:
            return False
    
    async def get_chain_of_custody(self, evidence_id: str) -> Dict[str, Any]:
        """Retrieve chain of custody for evidence"""
        try:
            evidence_record = await self._query_evidence(evidence_id)
            
            if not evidence_record:
                return {
                    "evidence_id": evidence_id,
                    "chain_of_custody": [],
                    "status": "not_found"
                }
            
            # Extract chain of custody from evidence data
            custody_chain = evidence_record.get("data", {}).get("chain_of_custody", [])
            
            return {
                "evidence_id": evidence_id,
                "chain_of_custody": custody_chain,
                "total_transfers": len(custody_chain),
                "current_handler": custody_chain[-1]["handler"] if custody_chain else "unknown",
                "status": "retrieved"
            }
            
        except Exception as e:
            logger.error(f"Chain of custody retrieval failed: {e}")
            return {
                "evidence_id": evidence_id,
                "chain_of_custody": [],
                "status": "error",
                "error": str(e)
            }
    
    async def add_custody_transfer(self, evidence_id: str, handler: str, action: str, location: str = "unknown") -> Dict[str, Any]:
        """Add custody transfer to evidence chain"""
        try:
            # Create custody record
            custody_record = {
                "handler": handler,
                "action": action,
                "timestamp": datetime.utcnow().isoformat(),
                "location": location
            }
            
            # Update evidence on blockchain
            update_result = await self._update_evidence_custody(evidence_id, custody_record)
            
            return {
                "evidence_id": evidence_id,
                "custody_record": custody_record,
                "update_status": update_result["status"],
                "transaction_hash": update_result.get("transaction_hash"),
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Custody transfer failed: {e}")
            raise
    
    async def _update_evidence_custody(self, evidence_id: str, custody_record: Dict[str, Any]) -> Dict[str, Any]:
        """Update evidence custody information on blockchain"""
        if not self.connected:
            # Simulation mode
            return {
                "status": "success",
                "transaction_hash": f"0x_update_hash_{evidence_id}",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # In production: await self.fabric_client.submit_transaction({
        #     "contract_id": self.contract_id,
        #     "function": "addCustodyRecord",
        #     "arguments": [evidence_id, custody_record]
        # })
        
        return {"status": "simulation_success"}

# Export the main class
__all__ = ["EvidenceLogger"]