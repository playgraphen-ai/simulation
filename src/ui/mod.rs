//! UI layer: tool palette (road / zone painting), activity-duration sliders,
//! spawn button, HUD counters.
//!
//! Built with plain bevy_ui nodes (no egui dependency).

pub mod tools;
pub mod hud;
pub mod paint;
pub mod inspector;
pub mod perf;
pub mod menu;
pub mod speed;

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use crate::AppState;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<tools::ActiveTool>()
            .init_resource::<inspector::Selection>()
            .add_plugins(ExtractResourcePlugin::<inspector::Selection>::default())
            .init_resource::<menu::MenuSettings>()
            .add_systems(OnEnter(AppState::MainMenu), menu::setup_menu)
            .add_systems(Update, menu::handle_menu_actions.run_if(in_state(AppState::MainMenu)))
            .add_systems(OnEnter(AppState::InGame), (
                hud::setup_hud,
                tools::setup_toolbar,
                inspector::setup_inspector,
                perf::setup_perf_ui,
                speed::setup_speed_ui,
            ))
            .add_systems(Update, (
                tools::handle_tool_buttons,
                tools::handle_spawn_button,
                tools::handle_sliders,
                tools::toggle_toolbar,
                hud::update_hud,
                hud::update_money_hud,
                paint::paint_tick_system,
                inspector::handle_selection,
                inspector::update_inspector_ui,
                inspector::draw_destination_arrow,
                inspector::toggle_inspector,
                perf::update_perf_ui,
                speed::handle_speed_buttons,
                speed::update_speed_button_colors,
            ).run_if(in_state(AppState::InGame)));
    }
}
