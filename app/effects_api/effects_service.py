from .dto.development_dto import DevelopmentDTO, ContextDevelopmentDTO


class EffectsService:

    def __init__(self):
        pass

    async def calc_project_development(self, params: DevelopmentDTO):
        """
        Function calculates development only for project with blocksnet
        Args:
            params (DevelopmentDTO):
        Returns:
            --
        """

        pass

    async def calc_context_development(self, params: ContextDevelopmentDTO):
        """
        Function calculates development for context  with project with blocksnet
        Args:
            params (DevelopmentDTO):
        Returns:
            --
        """

        pass
