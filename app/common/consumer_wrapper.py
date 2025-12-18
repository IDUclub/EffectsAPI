from otteroad import KafkaConsumerSettings, KafkaConsumerService, BaseMessageHandler


class ConsumerWrapper:
    def __init__(self):
        self.consumer_settings = KafkaConsumerSettings.from_env()
        self.consumer_service = KafkaConsumerService(self.consumer_settings)
    
    def register_handler(self, handler: BaseMessageHandler):
        self.consumer_service.register_handler(handler)
    
    async def start(self, topics: list[str]):
        await self.consumer_service.add_worker(topics=topics).start()
