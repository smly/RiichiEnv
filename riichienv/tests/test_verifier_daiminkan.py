import sys
import os
import pytest
from unittest.mock import MagicMock

# Ensure we can import from scripts
# Assuming running from repo root
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from verify_gym_api_with_mjsoul import MjsoulEnvVerifier
from riichienv import RiichiEnv
from riichienv.action import Action, ActionType
import riichienv.convert as cvt

class TestVerifierSmartScan:
    def test_daiminkan_smart_scan(self):
        """
        Verify that _handle_chipenggang correctly selects available tiles from the hand
        even if they don't match the canonical IDs for the tile type.
        """
        # 1. Setup Verifier and Env
        verifier = MjsoulEnvVerifier(verbose=True)
        env = RiichiEnv(seed=42)
        env.reset()
        verifier.env = env
        
        # 2. Setup P0 hand with non-canonical 1z (East)
        # Canonical 1z is 108. IDs are 108, 109, 110, 111.
        # Let's say we have 109, 110, 111 in hand. 
        # The MJSoul log event usually implies "1z". Canonical conversion gives 108.
        # If we naively look for 108, it won't be there.
        # Smart scan should find 109, 110, 111.
        
        p0 = 0
        tid_1z_canonical = cvt.mjai_to_tid("E") # 108
        hand_tids = [109, 110, 111] # 3 Easts (missing 108)
        # Fill rest
        hand_tids += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 13 tiles total
        
        env.hands[p0] = sorted(hand_tids)
        env.current_player = p0 # Doesn't strictly matter for verifier logic check
        
        # Mock Observation
        obs_mock = MagicMock()
        obs_mock.hand = sorted(hand_tids)
        obs_mock.legal_actions.return_value = [Action(ActionType.DAIMINKAN, tile=tid_1z_canonical, consume_tiles=[])]
        
        verifier.obs_dict = {p0: obs_mock}
        
        # 3. Construct DAIMINKAN Event
        # MJSoul logs use strings like "1z".
        # DAIMINKAN event: type 2
        # Tiles: ["1z", "1z", "1z", "1z"] (one is target, 3 are consumed)
        # froms: [p0, p0, p0, p1] (Last one from p1 is target)
        event = {
            "data": {
                "seat": p0,
                "type": 2, # DAIMINKAN
                "tiles": ["1z", "1z", "1z", "1z"],
                "froms": [p0, p0, p0, 1]
            }
        }
        
        # 4. Mock env.step to capture the action passed
        captured_action = None
        def mock_step(action_dict):
            nonlocal captured_action
            captured_action = action_dict[p0]
            # Must return a dict of observations, Mock it
            return {p0: obs_mock}
        
        env.step = mock_step
        
        # 5. Run Handler
        verifier._handle_chipenggang(event, p0, obs_mock)
        
        # 6. Verify Action
        assert captured_action is not None
        assert captured_action.type == ActionType.DAIMINKAN
        assert captured_action.tile == tid_1z_canonical # Target 108 (implied from 1z)
        
        # CRITICAL: Verify consumed tiles are the ones actually in hand (109, 110, 111)
        # NOT just [108, 108, 108] or [108, 109, 110]
        expected_consumed = [109, 110, 111] 
        # logic might pick in order of appearance in MJSoul event which is 3 "1z"s.
        # Smart scan orders by index? No, it iterates mpsz list.
        # And finds first match in hand.
        # Hand is sorted: 0..9, 109, 110, 111.
        # 1st "1z": matches 109. Remove 109.
        # 2nd "1z": matches 110. Remove 110.
        # 3rd "1z": matches 111. Remove 111.
        # So consumed should contain 109, 110, 111.
        
        print("Captured Consumed:", captured_action.consume_tiles)
        assert sorted(captured_action.consume_tiles) == sorted(expected_consumed)
        
        print(">> Smart Scan verified successfully.")
