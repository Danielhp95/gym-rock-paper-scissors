from .mixed_strategy_agent import MixedStrategyAgent

rockAgent     = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent')
paperAgent    = MixedStrategyAgent(support_vector=[0, 1, 0], name='PaperAgent')
scissorsAgent = MixedStrategyAgent(support_vector=[0, 0, 1], name='ScissorsAgent')
randomAgent   = MixedStrategyAgent(support_vector=[1/3, 1/3, 1/3], name='RandomAgent')
