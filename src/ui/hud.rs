//! Top-left HUD with population and building counters.

use bevy::prelude::*;

use crate::sim::counters::SimCounters;
use crate::sim::GameTime;

#[derive(Component)]
pub struct HudText;

#[derive(Component)]
pub struct MoneyHud;

#[derive(Component)]
pub struct MoneyTooltip;

pub fn setup_hud(mut commands: Commands) {
    // Top-left HUD
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

    // Top-center Money display
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Auto,
            right: Val::Auto,
            top: Val::Px(12.),
            margin: UiRect::horizontal(Val::Auto),
            padding: UiRect::all(Val::Px(10.)),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0., 0., 0., 0.7)),
        Interaction::default(),
        MoneyHud,
    )).with_children(|p| {
        p.spawn((
            Text::new("$0"),
            TextFont { font_size: 24.0, ..default() },
            TextColor(Color::srgb(0.2, 1.0, 0.2)),
        ));
        
        // Tooltip (hidden by default)
        p.spawn((
            Node {
                display: Display::None,
                flex_direction: FlexDirection::Column,
                margin: UiRect::top(Val::Px(8.)),
                ..default()
            },
            MoneyTooltip,
        )).with_children(|tp| {
            tp.spawn((
                Text::new("Income Tax: $0"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::WHITE),
            ));
            tp.spawn((
                Text::new("Rent Tax: $0"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::WHITE),
            ));
            tp.spawn((
                Text::new("Shop Tax: $0"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
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

pub fn update_money_hud(
    counters: Res<SimCounters>,
    q_hud: Query<(&Interaction, &Children), With<MoneyHud>>,
    mut q_tooltip: Query<&mut Node, With<MoneyTooltip>>,
    mut q_text: Query<&mut Text>,
    q_children: Query<&Children>,
) {
    let Ok((interaction, children)) = q_hud.single() else { return; };
    
    // Update main money text
    if let Ok(mut text) = q_text.get_mut(children[0]) {
        let total = counters.tax_income_total + counters.tax_rent_total + counters.tax_consumption_total;
        text.0 = format!("${}", total);
    }

    // Handle tooltip visibility
    if let Ok(mut tooltip_node) = q_tooltip.single_mut() {
        tooltip_node.display = match interaction {
            Interaction::Hovered => Display::Flex,
            _ => Display::None,
        };
        
        // Update tooltip texts
        if let Ok(tooltip_children) = q_children.get(children[1]) {
            if let Ok(mut t1) = q_text.get_mut(tooltip_children[0]) {
                t1.0 = format!("Income Tax: ${}", counters.tax_income_total);
            }
            if let Ok(mut t2) = q_text.get_mut(tooltip_children[1]) {
                t2.0 = format!("Rent Tax: ${}", counters.tax_rent_total);
            }
            if let Ok(mut t3) = q_text.get_mut(tooltip_children[2]) {
                t3.0 = format!("Shop Tax: ${}", counters.tax_consumption_total);
            }
        }
    }
}
