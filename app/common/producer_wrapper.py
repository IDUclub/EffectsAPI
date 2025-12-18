from otteroad import KafkaProducerSettings, KafkaProducerClient


class ProducerWrapper:
    def __init__(self):
        self.producer_settings = KafkaProducerSettings.from_env()
        self.producer_service = KafkaProducerClient(self.producer_settings)
    
    async def start(self):
        await self.producer_service.start()
