try:
    import faust
except ImportError:
    pass

import numpy as np
from systemictau import systemic_tau
from systemictau.streaming.models import StreamPayload
from pydantic import ValidationError
from prometheus_client import Counter
from systemictau.config import settings

# Observability Metrics
anomalies_detected = Counter('tau_anomalies_detected_total', 'Total number of systemic anomalies detected')

# Ensure this runs gracefully if faust is missing (e.g. standard install)
if 'faust' in globals():
    kafka_broker = settings.kafka_broker
    
    app = faust.App(
        'systemictau-stream-processor',
        broker=kafka_broker,
        value_serializer='json',
        store='rocksdb://', # Enables durable local state recovery
        topic_partitions=4, # Horizontal scaling
    )
    
    # Kafka Topic definitions
    raw_data_topic = app.topic('sys.raw_data', partitions=4)
    transitions_topic = app.topic('sys.transitions', partitions=4)
    dlq_topic = app.topic('sys.dlq', partitions=4)
    
    # Stateful table to keep a rolling window
    window_state = app.Table('sliding_windows', default=list)
    
    @app.agent(raw_data_topic)
    async def process_stream(stream):
        async for payload in stream:
            try:
                # 1. Strict Payload Validation
                validated_data = StreamPayload(**payload)
            except ValidationError as e:
                print(f"[FAUST] Validation Error. Routing to DLQ: {e}")
                await dlq_topic.send(value={"error": str(e), "raw_payload": payload})
                continue
                
            tenant_id = validated_data.tenant_id
            new_data_point = validated_data.vector
            
            # Append to rolling window (simplified logic)
            window = window_state[tenant_id]
            window.append(new_data_point)
            
            # Maintain fixed window size (e.g., 13)
            if len(window) > 13:
                window.pop(0)
            
            window_state[tenant_id] = window
            
            # Compute Systemic Tau if we have enough data
            if len(window) == 13:
                X = np.array(window)
                # Compute Tau
                res = systemic_tau(X, window_size=13, engine="numba")
                latest_tau = res.taus_global[-1]
                
                # Check anomaly threshold for Critical Mass (e.g. > 0.8)
                if latest_tau > 0.8:
                    anomalies_detected.inc()
                    await transitions_topic.send(value={
                        "tenant_id": tenant_id,
                        "tau": float(latest_tau),
                        "message": "Potential Critical Mass Detected"
                    })
