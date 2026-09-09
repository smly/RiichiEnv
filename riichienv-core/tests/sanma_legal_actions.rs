//! Regressions from comparing legal actions with Mahjong Soul's operation menus.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state_3p::{GameState3P, legal_actions::GameState3PLegalActions};

fn rules() -> [GameRule; 2] {
    [GameRule::default_mjsoul(), GameRule::default_tenhou()]
}

fn offered(state: &GameState3P, seat: u8, kind: ActionType) -> Vec<Action> {
    state
        ._get_legal_actions_internal(seat)
        .into_iter()
        .filter(|action| action.action_type == kind)
        .collect()
}

fn act(state: &mut GameState3P, seat: u8, kind: ActionType) {
    let action = offered(state, seat, kind).into_iter().next().unwrap();
    state.step(&HashMap::from([(seat, action)]));
    assert!(state.last_error.is_none());
}

fn discard(state: &mut GameState3P, seat: u8, tile34: u8) {
    let action = offered(state, seat, ActionType::Discard)
        .into_iter()
        .find(|action| action.tile.unwrap() / 4 == tile34)
        .unwrap();
    state.step(&HashMap::from([(seat, action)]));
    assert!(state.last_error.is_none());
}

fn pass_all(state: &mut GameState3P) {
    let actions = state
        .active_players
        .iter()
        .map(|&seat| {
            (
                seat,
                Action::new(ActionType::Pass, None, vec![], Some(seat)),
            )
        })
        .collect();
    state.step(&actions);
    assert!(state.last_error.is_none());
}

fn north_hand(rule: GameRule, drawn: u8) -> GameState3P {
    let mut state = GameState3P::new(3, false, Some(233), 0, rule);
    // 123456789p12sNN waits on 3s; drawn is a separate 14th tile.
    state.players[0].hand = vec![36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 120, 121];
    state.players[0].hand.push(drawn);
    state.current_player = 0;
    state.active_players = vec![0];
    state.phase = Phase::WaitAct;
    state.drawn_tile = Some(drawn);
    state.needs_tsumo = false;
    state.is_first_turn = false;
    // Other players cannot ron North; fix the replacement draw to Haku.
    state.players[1].hand.clear();
    state.players[2].hand.clear();
    state.wall.tiles[0] = 124;
    state
}

#[test]
fn riichi_requires_a_remaining_draw_cycle() {
    for rule in rules() {
        for left in 0..=4 {
            let mut state = north_hand(rule, 108);
            state.wall.drawable_count = left;
            assert_eq!(
                !offered(&state, 0, ActionType::Riichi).is_empty(),
                left >= 3,
                "sanma riichi with {left} tiles remaining"
            );
        }
    }
}

#[test]
fn riichi_at_three_remaining_tiles_can_be_declared() {
    for rule in rules() {
        let mut state = north_hand(rule, 108);
        state.wall.drawable_count = 3;
        act(&mut state, 0, ActionType::Riichi);
        assert!(state.players[0].riichi_stage);
        discard(&mut state, 0, 27);
        assert!(state.players[0].riichi_declared);
        assert_eq!(state.players[0].score, 34000);
        assert_eq!(state.riichi_sticks, 1);
    }
}

#[test]
fn before_riichi_all_held_norths_can_be_extracted() {
    for rule in rules() {
        let state = north_hand(rule, 108);
        let tiles: Vec<_> = offered(&state, 0, ActionType::Kita)
            .iter()
            .map(|a| a.tile.unwrap())
            .collect();
        assert_eq!(tiles, vec![120, 121]);
    }
}

#[test]
fn riichi_cannot_extract_held_north_after_other_draw() {
    for rule in rules() {
        let mut state = north_hand(rule, 108);
        state.players[0].riichi_declared = true;
        assert!(offered(&state, 0, ActionType::Kita).is_empty());
        assert_eq!(offered(&state, 0, ActionType::Discard)[0].tile, Some(108));
    }
}

#[test]
fn riichi_kita_only_offers_the_drawn_physical_copy() {
    for rule in rules() {
        let mut state = north_hand(rule, 122);
        state.players[0].riichi_declared = true;
        let tiles: Vec<_> = offered(&state, 0, ActionType::Kita)
            .iter()
            .map(|a| a.tile.unwrap())
            .collect();
        assert_eq!(tiles, vec![122]);
    }
}

#[test]
fn generic_kita_after_riichi_preserves_the_original_hand() {
    for rule in rules() {
        let mut state = north_hand(rule, 122);
        state.players[0].riichi_declared = true;
        let original = state.players[0].hand[..13].to_vec();
        state.step(&HashMap::from([(
            0,
            Action::new(ActionType::Kita, None, vec![], Some(0)),
        )]));
        assert!(state.last_error.is_none());
        assert_eq!(state.players[0].kita_tiles, vec![122]);
        assert_eq!(state.players[0].hand[..13], original);
        assert_eq!(state.drawn_tile, Some(124));
    }
}

#[test]
fn kita_is_not_offered_between_riichi_declaration_and_discard() {
    for rule in rules() {
        let mut state = north_hand(rule, 122);
        act(&mut state, 0, ActionType::Riichi);
        assert!(state.players[0].riichi_stage);
        assert!(offered(&state, 0, ActionType::Kita).is_empty());
        assert!(!offered(&state, 0, ActionType::Discard).is_empty());
    }
}

#[test]
fn kita_requires_a_replacement_draw() {
    for rule in rules() {
        let mut state = north_hand(rule, 122);
        state.wall.drawable_count = 0;
        assert!(offered(&state, 0, ActionType::Kita).is_empty());
    }
}

// Exact wall of round 5 (zero-based) in
// 251203-7767b176-135e-4b16-a6ce-cb0ca6c1676a, with no profile data.
const REPORTED_WALL: [u8; 108] = [
    124, 132, 0, 133, 68, 104, 60, 128, 89, 92, 44, 120, 90, 36, 80, 129, 88, 96, 48, 116, 121, 49,
    40, 61, 134, 81, 122, 112, 105, 64, 62, 125, 117, 69, 65, 97, 100, 108, 118, 66, 76, 101, 70,
    91, 130, 53, 98, 56, 50, 77, 37, 102, 41, 113, 32, 54, 67, 45, 114, 84, 106, 71, 85, 33, 72,
    86, 1, 82, 119, 99, 131, 55, 63, 73, 93, 126, 94, 74, 2, 52, 57, 109, 87, 110, 83, 103, 3, 46,
    47, 42, 78, 79, 43, 58, 123, 34, 135, 111, 107, 95, 115, 59, 127, 38, 51, 39, 75, 35,
];

fn reported_position(rule: GameRule) -> GameState3P {
    let mut state = GameState3P::new(3, false, Some(233), 0, rule);
    state._initialize_round(
        0,
        1,
        0,
        0,
        Some(REPORTED_WALL.to_vec()),
        Some(vec![64000, 20600, 20400]),
    );
    state
}

#[test]
fn no_yaku_kita_wait_blocks_later_same_turn_ron() {
    for rule in rules() {
        let mut state = reported_position(rule);
        // West's W/N shanpon has a yaku on W but none on N.
        assert!(!state.players[2].missed_agari_doujun);
        act(&mut state, 0, ActionType::Kita);
        assert!(state.players[2].missed_agari_doujun);
        assert!(!state.players[2].missed_agari_riichi);
        assert!(
            !state.players[1].missed_agari_doujun,
            "a non-waiter is unaffected"
        );
        discard(&mut state, 0, 0); // Dealer's 1m.
        discard(&mut state, 1, 29); // South's West.
        assert!(offered(&state, 2, ActionType::Ron).is_empty());
        assert!(!offered(&state, 2, ActionType::Pon).is_empty());
    }
}

#[test]
fn temporary_kita_furiten_clears_after_the_players_turn() {
    for rule in rules() {
        let mut state = reported_position(rule);
        act(&mut state, 0, ActionType::Kita);
        assert!(state.players[2].missed_agari_doujun);
        discard(&mut state, 0, 0);
        discard(&mut state, 1, 29);
        pass_all(&mut state);
        assert_eq!(state.current_player, 2);
        let drawn = state.drawn_tile.unwrap();
        discard(&mut state, 2, drawn / 4);
        assert!(!state.players[2].missed_agari_doujun);
        let (claims, _) = state._get_claim_actions_for_player(2, 1, 116);
        assert!(claims.iter().any(|a| a.action_type == ActionType::Ron));
    }
}

#[test]
fn kita_with_a_yaku_can_still_be_ronned() {
    for rule in rules() {
        let mut state = reported_position(rule);
        state.players[2].riichi_declared = true;
        act(&mut state, 0, ActionType::Kita);
        assert_eq!(state.phase, Phase::WaitResponse);
        assert!(!state.players[2].missed_agari_doujun);
        act(&mut state, 2, ActionType::Ron);
        assert!(state.is_done);
        assert!(state.win_results.contains_key(&2));
    }
}

#[test]
fn declining_kita_ron_after_riichi_remains_permanent_furiten() {
    for rule in rules() {
        let mut state = reported_position(rule);
        state.players[2].riichi_declared = true;
        act(&mut state, 0, ActionType::Kita);
        pass_all(&mut state);
        assert!(state.players[2].missed_agari_doujun);
        assert!(state.players[2].missed_agari_riichi);
        discard(&mut state, 0, 0);
        discard(&mut state, 1, 29);
        assert!(offered(&state, 2, ActionType::Ron).is_empty());
    }
}
