use bevy::prelude::*;
use crate::AppState;

#[derive(Resource)]
pub struct MenuSettings {
    pub grid_x: u32,
    pub grid_y: u32,
    pub population: u32,
}

impl Default for MenuSettings {
    fn default() -> Self {
        Self { grid_x: 4, grid_y: 4, population: 100 }
    }
}

#[derive(Component)]
pub struct MainMenuRoot;

#[derive(Component)]
pub enum MenuAction {
    Small,
    Medium,
    Large,
}

pub fn setup_menu(mut commands: Commands) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(20.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.8)),
        MainMenuRoot,
    )).with_children(|p| {
        p.spawn((
            Text::new("REAL CITY"),
            TextFont { font_size: 60.0, ..default() },
            TextColor(Color::WHITE),
            Node { margin: UiRect::bottom(Val::Px(40.0)), ..default() },
        ));

        spawn_btn(p, "Small City (3x3 blocks, 50 pop)", MenuAction::Small);
        spawn_btn(p, "Medium City (6x5 blocks, 300 pop)", MenuAction::Medium);
        spawn_btn(p, "Large City (20x16 blocks, 1000 pop)", MenuAction::Large);
    });
}

fn spawn_btn(p: &mut ChildSpawnerCommands, text: &str, action: MenuAction) {
    p.spawn((
        Button,
        Node {
            width: Val::Px(400.0),
            height: Val::Px(60.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgb(0.2, 0.3, 0.4)),
        action,
    )).with_children(|b| {
        b.spawn((
            Text::new(text),
            TextFont { font_size: 24.0, ..default() },
            TextColor(Color::WHITE),
        ));
    });
}

pub fn handle_menu_actions(
    mut commands: Commands,
    mut q: Query<(&Interaction, &MenuAction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
    mut next_state: ResMut<NextState<AppState>>,
    mut settings: ResMut<MenuSettings>,
    root: Query<Entity, With<MainMenuRoot>>,
) {
    for (interaction, action, mut bg) in &mut q {
        match *interaction {
            Interaction::Pressed => {
                match action {
                    MenuAction::Small => { settings.grid_x = 3; settings.grid_y = 3; settings.population = 50; }
                    MenuAction::Medium => { settings.grid_x = 6; settings.grid_y = 5; settings.population = 300; }
                    MenuAction::Large => { settings.grid_x = 20; settings.grid_y = 16; settings.population = 1000; }
                }
                if let Ok(e) = root.single() {
                    commands.entity(e).despawn();
                }
                next_state.set(AppState::InGame);
            }
            Interaction::Hovered => bg.0 = Color::srgb(0.3, 0.4, 0.5),
            Interaction::None => bg.0 = Color::srgb(0.2, 0.3, 0.4),
        }
    }
}