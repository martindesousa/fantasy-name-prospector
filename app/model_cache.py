import tensorflow as tf
import gc
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

class ModelCache:
    """Cache for managing models per user with automatic expiration."""
    
    def __init__(self, max_size_per_user=2, max_age_hours=1):
        self.max_size_per_user = max_size_per_user
        self.max_age_hours = max_age_hours
        self._cache = {}  # {user_id: OrderedDict of {model_name: model_data}}
        self._cache_timestamps = {}  # {user_id: datetime of last access}
        self._lock = Lock()
    
    def _cleanup_stale_users(self):
        """Remove caches for users who haven't been active recently."""
        if not self._cache_timestamps:
            return
        
        now = datetime.now()
        cutoff = now - timedelta(hours=self.max_age_hours)
        
        stale_users = [
            user_id for user_id, last_access in self._cache_timestamps.items()
            if last_access < cutoff
        ]
        
        for user_id in stale_users:
            if user_id in self._cache:
                user_cache = self._cache[user_id]
                
                # Delete all models for this user
                for model_data in user_cache.values():
                    if model_data is not None and isinstance(model_data, tuple):
                        model = model_data[0] if len(model_data) > 0 else None
                        if model is not None:
                            del model
                        del model_data
                
                del self._cache[user_id]
                del self._cache_timestamps[user_id]
        
        if stale_users:
            tf.keras.backend.clear_session()
            gc.collect()
    
    def _get_user_cache(self, user_id):
        """Get or create cache for a specific user."""
        if user_id not in self._cache:
            self._cache[user_id] = OrderedDict()
        return self._cache[user_id]
    
    def _evict_oldest_for_user(self, user_id):
        """Remove the oldest model from a user's cache."""
        user_cache = self._get_user_cache(user_id)
        
        if len(user_cache) >= self.max_size_per_user:
            # Remove the oldest entry (first item in OrderedDict)
            oldest_key, model_data = user_cache.popitem(last=False)
            
            # Explicitly delete the model to free memory
            if model_data is not None and isinstance(model_data, tuple):
                # Extract model from tuple and delete it
                model = model_data[0] if len(model_data) > 0 else None
                if model is not None:
                    del model
                # Delete the entire tuple
                del model_data
            
            # Clear TensorFlow session and garbage collect
            tf.keras.backend.clear_session()
            gc.collect()
    
    def get(self, user_id, model_name):
        """Retrieve a cached model for a user."""
        with self._lock:
            # Periodically cleanup stale caches (every ~20 cache lookups)
            import random
            if random.randint(1, 20) == 1:
                self._cleanup_stale_users()
            
            user_cache = self._get_user_cache(user_id)
            
            if model_name in user_cache:
                # Update last access time
                self._cache_timestamps[user_id] = datetime.now()
                # Move to end to mark as recently used
                user_cache.move_to_end(model_name)
                return user_cache[model_name]
            
            return None
    
    def put(self, user_id, model_name, model_data):
        """Store a model in the cache for a user."""
        with self._lock:
            user_cache = self._get_user_cache(user_id)
            
            # Update last access time
            self._cache_timestamps[user_id] = datetime.now()
            
            # Evict oldest if cache is full
            self._evict_oldest_for_user(user_id)
            
            # Add new model data
            user_cache[model_name] = model_data
    
    def clear_user_cache(self, user_id):
        """Clear all cached models for a specific user."""
        with self._lock:
            if user_id in self._cache:
                user_cache = self._cache[user_id]
                
                # Delete all models for this user
                for model_data in user_cache.values():
                    if model_data is not None and isinstance(model_data, tuple):
                        # Extract model from tuple and delete it
                        model = model_data[0] if len(model_data) > 0 else None
                        if model is not None:
                            del model
                        del model_data
                
                del self._cache[user_id]
                tf.keras.backend.clear_session()
                gc.collect()
    
    def clear_all(self):
        """Clear all cached models for all users."""
        with self._lock:
            for user_id in list(self._cache.keys()):
                user_cache = self._cache[user_id]
                
                # Delete all models
                for model_data in user_cache.values():
                    if model_data is not None and isinstance(model_data, tuple):
                        # Extract model from tuple and delete it
                        model = model_data[0] if len(model_data) > 0 else None
                        if model is not None:
                            del model
                        del model_data
            
            self._cache.clear()
            tf.keras.backend.clear_session()
            gc.collect()
    
    def get_cache_stats(self):
        """Get statistics about current cache usage."""
        with self._lock:
            now = datetime.now()
            stats = {}
            for user_id, user_cache in self._cache.items():
                last_access = self._cache_timestamps.get(user_id)
                hours_since_access = None
                if last_access:
                    hours_since_access = (now - last_access).total_seconds() / 3600
                
                stats[user_id] = {
                    'model_count': len(user_cache),
                    'models': list(user_cache.keys()),
                    'last_access_hours_ago': round(hours_since_access, 2) if hours_since_access else None
                }
            return stats


# Global cache instance (expires after 24 hours of inactivity)
_model_cache = ModelCache(max_size_per_user=2, max_age_hours=24)

# Global cache for trigram data (this can be shared across users)
_trigram_endings = {}
_trigram_lock = Lock()


def get_model_from_cache(user_id, model_name):
    """Retrieve a model from the cache."""
    return _model_cache.get(user_id, model_name)


def put_model_in_cache(user_id, model_name, model_data):
    """Store a model in the cache."""
    _model_cache.put(user_id, model_name, model_data)


def clear_user_model_cache(user_id):
    """Clear all cached models for a specific user."""
    _model_cache.clear_user_cache(user_id)


def clear_all_model_caches():
    """Clear all cached models."""
    _model_cache.clear_all()


def get_cache_statistics():
    """Get cache usage statistics."""
    return _model_cache.get_cache_stats()


def get_trigram_cache(cache_key):
    """Thread-safe retrieval of trigram endings."""
    with _trigram_lock:
        return _trigram_endings.get(cache_key)


def put_trigram_cache(cache_key, trigrams):
    """Thread-safe storage of trigram endings."""
    with _trigram_lock:
        _trigram_endings[cache_key] = trigrams