import tensorflow as tf
import gc
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

class ModelCache:
    """Thread-safe cache with model sharing across users."""
    
    def __init__(self, max_models_per_user=2, max_age_hours=24):
        self.max_models_per_user = max_models_per_user
        self.max_age_hours = max_age_hours
        
        # Shared model storage - models stored once, referenced by multiple users
        self._models = {}  # {model_key: model_data_tuple}
        self._model_ref_count = {}  # {model_key: number of users using it}
        
        # Per-user tracking - which models each user is using
        self._user_models = {}  # {user_id: OrderedDict of model_keys}
        self._user_timestamps = {}  # {user_id: datetime of last access}
        
        self._lock = Lock()
        self._cleanup_counter = 0
    
    def _make_model_key(self, user_id, model_name):
        """Create cache key. Custom models are user-specific, built-in models are shared."""
        if model_name.startswith('custom'):
            # Custom models are per-user (not shared)
            return f"{user_id}:{model_name}"
        else:
            # Built-in models are shared across all users
            return f"shared:{model_name}"
    
    def _cleanup_stale_users(self):
        """Remove inactive users and unreferenced models."""
        if not self._user_timestamps:
            return
        
        now = datetime.now()
        cutoff = now - timedelta(hours=self.max_age_hours)
        
        stale_users = [
            user_id for user_id, last_access in self._user_timestamps.items()
            if last_access < cutoff
        ]
        
        for user_id in stale_users:
            self._remove_user(user_id)
        
        if stale_users:
            tf.keras.backend.clear_session()
            gc.collect()
    
    def _remove_user(self, user_id):
        """Remove a user and decrement reference counts for their models."""
        if user_id not in self._user_models:
            return
        
        user_model_keys = self._user_models[user_id]
        
        # Decrement reference count for each model this user was using
        for model_key in user_model_keys:
            if model_key in self._model_ref_count:
                self._model_ref_count[model_key] -= 1
                
                # If no users reference this model, delete it
                if self._model_ref_count[model_key] <= 0:
                    self._delete_model(model_key)
        
        # Remove user tracking
        del self._user_models[user_id]
        if user_id in self._user_timestamps:
            del self._user_timestamps[user_id]
    
    def _delete_model(self, model_key):
        """Delete a model from memory."""
        if model_key in self._models:
            model_data = self._models[model_key]
            
            if model_data is not None and isinstance(model_data, tuple):
                model = model_data[0] if len(model_data) > 0 else None
                if model is not None:
                    del model
                del model_data
            
            del self._models[model_key]
            if model_key in self._model_ref_count:
                del self._model_ref_count[model_key]
    
    def _evict_oldest_model_for_user(self, user_id):
        """Remove the oldest model from a user's list."""
        if user_id not in self._user_models:
            return
        
        user_model_keys = self._user_models[user_id]
        
        if len(user_model_keys) >= self.max_models_per_user:
            # Remove oldest (first item in OrderedDict)
            oldest_key, _ = user_model_keys.popitem(last=False)
            
            # Decrement reference count
            if oldest_key in self._model_ref_count:
                self._model_ref_count[oldest_key] -= 1
                
                # Delete model if no one is using it
                if self._model_ref_count[oldest_key] <= 0:
                    self._delete_model(oldest_key)
    
    def get(self, user_id, model_name):
        """Retrieve a cached model."""
        with self._lock:
            # Periodic cleanup
            self._cleanup_counter += 1
            if self._cleanup_counter >= 20:
                self._cleanup_counter = 0
                self._cleanup_stale_users()
            
            model_key = self._make_model_key(user_id, model_name)
            
            # Check if model exists in cache
            if model_key not in self._models:
                return None
            
            # Initialize user tracking if needed
            if user_id not in self._user_models:
                self._user_models[user_id] = OrderedDict()
            
            # Update user's access time and model order
            self._user_timestamps[user_id] = datetime.now()
            
            # Mark this model as recently used by this user
            if model_key in self._user_models[user_id]:
                self._user_models[user_id].move_to_end(model_key)
            else:
                # User is accessing this model for the first time
                self._user_models[user_id][model_key] = True
                self._model_ref_count[model_key] = self._model_ref_count.get(model_key, 0) + 1
            
            return self._models[model_key]
    
    def put(self, user_id, model_name, model_data):
        """Store a model in the cache for a user."""
        with self._lock:
            model_key = self._make_model_key(user_id, model_name)
            
            # Initialize user tracking
            if user_id not in self._user_models:
                self._user_models[user_id] = OrderedDict()
            
            # Update last access time
            self._user_timestamps[user_id] = datetime.now()
            
            # Evict oldest model if user has too many
            self._evict_oldest_model_for_user(user_id)
            
            # Store the model (if not already stored)
            if model_key not in self._models:
                self._models[model_key] = model_data
                self._model_ref_count[model_key] = 0
            
            # Add to user's model list if not already there
            if model_key not in self._user_models[user_id]:
                self._user_models[user_id][model_key] = True
                self._model_ref_count[model_key] += 1
            else:
                # Move to end (most recently used)
                self._user_models[user_id].move_to_end(model_key)
    
    def clear_user_cache(self, user_id):
        """Clear all cached models for a specific user."""
        with self._lock:
            self._remove_user(user_id)
            tf.keras.backend.clear_session()
            gc.collect()
    
    def clear_all(self):
        """Clear all cached models."""
        with self._lock:
            # Delete all models
            for model_key in list(self._models.keys()):
                self._delete_model(model_key)
            
            self._user_models.clear()
            self._user_timestamps.clear()
            self._model_ref_count.clear()
            
            tf.keras.backend.clear_session()
            gc.collect()
    
    def get_cache_stats(self):
        """Get statistics about current cache usage."""
        with self._lock:
            now = datetime.now()
            
            stats = {
                'total_models_in_memory': len(self._models),
                'total_active_users': len(self._user_models),
                'models': {},
                'users': {}
            }
            
            # Model statistics
            for model_key, ref_count in self._model_ref_count.items():
                stats['models'][model_key] = {
                    'reference_count': ref_count,
                    'is_shared': model_key.startswith('shared:')
                }
            
            # User statistics
            for user_id, user_models in self._user_models.items():
                last_access = self._user_timestamps.get(user_id)
                hours_since_access = None
                if last_access:
                    hours_since_access = (now - last_access).total_seconds() / 3600
                
                stats['users'][user_id] = {
                    'model_count': len(user_models),
                    'models': list(user_models.keys()),
                    'last_access_hours_ago': round(hours_since_access, 2) if hours_since_access else None
                }
            
            return stats


# Global cache instance (expires after 24 hours of inactivity)
_model_cache = ModelCache(max_models_per_user=2, max_age_hours=24)

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