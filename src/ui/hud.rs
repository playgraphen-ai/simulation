//! Top-left HUD with population and building counters.

use bevy::prelude::*;

use crate::sim::counters::SimCounters;
use crate::sim::GameTime;

#[derive(Component)]
pub struct HudText;

pub fn setup_hud(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.),
            top: Val::Px(12.),
            padding: UiRect::all(Val::Px(8.)),
            ..default()
        },
        BackgroundColor(Color::srgba(0., 0., 0., 0.55)),
    )).with_children(|p| {
        p.spawn((
            Text::new("Real City"),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::WHITE),
            HudText,
        ));
    });
}

pub fn update_hud(
    counters: Res<SimCounters>,
    game_time: Res<GameTime>,
    mut q: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = q.single_mut() else { return; };
    
    let (d, h, m) = game_time.simulation_time();
    let real_m = (game_time.real_elapsed_secs / 60.0) as u32;
    let real_s = (game_time.real_elapsed_secs % 60.0) as u32;

    text.0 = format!(
        "Time: Day {}, {:02}:{:02} (Real: {:02}:{:02})\n\
        People: {} (Bankrupt: {})\nTravelling (Cars): {}\nAt Home: {}\nAt Work: {}\nShopping: {}\nResidential: {}\nOffice: {}\nShop: {}\nRoads: {}\nAbandoned: {}",
        d, h, m, real_m, real_s,
        counters.people, counters.bankrupt,
        counters.cars,
        counters.res_occupants,
        counters.office_occupants,
        counters.shop_occupants,
        counters.residential,
        counters.offices,
        counters.shops,
        counters.road_segments,
        counters.destroyed_buildings,
    );
}
