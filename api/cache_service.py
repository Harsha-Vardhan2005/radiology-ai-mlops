# api/cache_service.py
import redis
import json
import hashlib
import os
import logging
from typing import Optional, Tuple, Any
import pickle
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        """Initialize Redis connection"""
        redis_host = os.getenv('REDIS_HOST', 'redis')  # 'redis' from docker-compose service name
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,  # We'll handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            self.connected = True
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e}. Falling back to memory cache.")
            self.redis_client = None
            self.connected = False
            # Fallback to in-memory cache
            self.memory_cache = {}

    def _generate_image_hash(self, image_path: str) -> str:
        """Generate unique hash for image file"""
        with open(image_path, 'rb') as f:
            file_content = f.read()
        return hashlib.md5(file_content).hexdigest()

    def _generate_chat_hash(self, message: str, diagnosis: str, confidence: float) -> str:
        """Generate unique hash for chat context"""
        context = f"{message}|{diagnosis}|{confidence:.4f}"
        return hashlib.md5(context.encode()).hexdigest()

    def cache_prediction(self, image_path: str, prediction: str, confidence: float, ttl: int = 3600):
        """Cache prediction result for an image"""
        try:
            image_hash = self._generate_image_hash(image_path)
            cache_key = f"prediction:{image_hash}"
            
            cache_data = {
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': int(os.times().elapsed)
            }
            
            if self.connected:
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(cache_data)
                )
                logger.info(f"🔄 Cached prediction for image hash: {image_hash[:8]}...")
            else:
                # Fallback to memory
                self.memory_cache[cache_key] = cache_data
                
        except Exception as e:
            logger.error(f"Failed to cache prediction: {e}")

    def get_cached_prediction(self, image_path: str) -> Optional[Tuple[str, float]]:
        """Get cached prediction for an image"""
        try:
            image_hash = self._generate_image_hash(image_path)
            cache_key = f"prediction:{image_hash}"
            
            if self.connected:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached.decode())
                    logger.info(f"✅ Cache HIT for image: {image_hash[:8]}...")
                    return data['prediction'], data['confidence']
            else:
                # Fallback to memory
                if cache_key in self.memory_cache:
                    data = self.memory_cache[cache_key]
                    logger.info(f"✅ Memory cache HIT for image: {image_hash[:8]}...")
                    return data['prediction'], data['confidence']
                    
            logger.info(f"❌ Cache MISS for image: {image_hash[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached prediction: {e}")
            return None

    def cache_chat_response(self, message: str, diagnosis: str, confidence: float, 
                          llm_response: str, ttl: int = 1800):
        """Cache chat response for similar medical questions"""
        try:
            chat_hash = self._generate_chat_hash(message, diagnosis, confidence)
            cache_key = f"chat:{chat_hash}"
            
            cache_data = {
                'response': llm_response,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'timestamp': int(os.times().elapsed)
            }
            
            if self.connected:
                self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
                logger.info(f"🔄 Cached chat response: {chat_hash[:8]}...")
            else:
                self.memory_cache[cache_key] = cache_data
                
        except Exception as e:
            logger.error(f"Failed to cache chat response: {e}")

    def get_cached_chat_response(self, message: str, diagnosis: str, confidence: float) -> Optional[str]:
        """Get cached chat response"""
        try:
            chat_hash = self._generate_chat_hash(message, diagnosis, confidence)
            cache_key = f"chat:{chat_hash}"
            
            if self.connected:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached.decode())
                    logger.info(f"✅ Chat cache HIT: {chat_hash[:8]}...")
                    return data['response']
            else:
                if cache_key in self.memory_cache:
                    data = self.memory_cache[cache_key]
                    logger.info(f"✅ Chat memory cache HIT: {chat_hash[:8]}...")
                    return data['response']
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached chat response: {e}")
            return None

    def store_session(self, session_id: str, session_data: dict, ttl: int = 7200):
        """Store session data (replaces in-memory current_session)"""
        try:
            cache_key = f"session:{session_id}"
            
            if self.connected:
                self.redis_client.setex(cache_key, ttl, json.dumps(session_data))
            else:
                self.memory_cache[cache_key] = session_data
                
        except Exception as e:
            logger.error(f"Failed to store session: {e}")

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        try:
            cache_key = f"session:{session_id}"
            
            if self.connected:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached.decode())
            else:
                return self.memory_cache.get(cache_key)
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def clear_session(self, session_id: str):
        """Clear session data"""
        try:
            cache_key = f"session:{session_id}"
            
            if self.connected:
                self.redis_client.delete(cache_key)
            else:
                self.memory_cache.pop(cache_key, None)
                
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            if self.connected:
                info = self.redis_client.info()
                return {
                    'connected': True,
                    'used_memory': info.get('used_memory_human', 'Unknown'),
                    'total_connections': info.get('total_connections_received', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            else:
                return {
                    'connected': False,
                    'memory_cache_keys': len(self.memory_cache),
                    'fallback_mode': True
                }
        except Exception as e:
            return {'error': str(e)}

# Global cache instance
cache = RedisCache()