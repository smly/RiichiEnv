from riichienv import GameType, RiichiEnv


class TestGameModes:
    def test_tonpusen_support(self):
        # Should not raise NotImplementedError anymore
        RiichiEnv(game_type=GameType.YON_TONPUSEN)

    def test_initialization_params(self):
        # East Round, 30000 start, 1 kyotaku, 2 honba
        custom_scores = [30000, 30000, 30000, 30000]
        env = RiichiEnv(
            game_type=GameType.YON_IKKYOKU,
            round_wind=0,  # East,
        )
        env.reset(
            scores=custom_scores,
            kyotaku=1,
            honba=2,
        )

        # Check internal state
        assert env.scores() == custom_scores
        assert env.riichi_sticks == 1
        assert env.game_type == GameType.YON_IKKYOKU

        # Check start_kyoku event
        start_kyoku = env.mjai_log[1]  # 0 is start_game, 1 is start_kyoku
        assert start_kyoku["type"] == "start_kyoku"
        assert start_kyoku["bakaze"] == "E"
        assert start_kyoku["honba"] == 2
        assert start_kyoku["kyotaku"] == 1

    def test_south_round_wind(self):
        # South Round
        env = RiichiEnv(
            game_type=GameType.YON_IKKYOKU,
            round_wind=1,  # South
        )
        env.reset()

        start_kyoku = env.mjai_log[1]
        assert start_kyoku["bakaze"] == "S"
