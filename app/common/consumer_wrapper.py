from otteroad import KafkaConsumerSettings, KafkaConsumerService, BaseMessageHandler


class ConsumerWrapper:
    def __init__(self):
        self.consumer_settings = KafkaConsumerSettings.from_env()
        self.consumer_service = KafkaConsumerService(self.consumer_settings)

    def register_handler(self, handler: BaseMessageHandler) -> None:
        self.consumer_service.register_handler(handler)

    async def start(self, topics: list[str]):
        self.consumer_service.add_worker(topics=topics)
        await self.consumer_service.start()

    async def stop(self) -> None:
        """Gracefully stop all consumer workers."""
        await self.consumer_service.stop()
