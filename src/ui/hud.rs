//! Top-left HUD with population and building counters.

use bevy::prelude::*;

use crate::sim::counters::SimCounters;

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
    mut q: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = q.single_mut() else { return; };
    text.0 = format!(
        "People: {}\nCars: {}\nResidential: {}\nOffice: {}\nShop: {}\nRoads: {}\nAbandoned: {}",
        counters.people,
        counters.cars,
        counters.residential,
        counters.offices,
        counters.shops,
        counters.road_segments,
        counters.destroyed_buildings,
    );
}
