-- Make colum names more natural

-- drop index 
DROP INDEX games_rotowire_entries_game_id_index;
DROP INDEX games_rotowire_entries_rotowire_entry_id_index;

-- drop column
ALTER TABLE person_stats DROP COLUMN start_position;

-- modify table team_in_game
CREATE TABLE new_team_in_game AS
(
    SELECT summary_id,
		   home_city AS city, 
           home_conference AS conference, 
           home_division AS division,
		   TRUE AS home
    FROM team_in_game

    UNION ALL

    SELECT summary_id,
		   vis_city AS city, 
           vis_conference AS conference, 
           vis_division AS division,
		   FALSE AS home
    FROM team_in_game
);

CREATE TABLE game_with_cities AS 
SELECT game_summary.summary_id, 
       home_team.city as home_city, 
       vis_team.city as vis_city,
	   home_team.team_id as home_team_id,
	   vis_team.team_id as vis_team_id
FROM game_summary
JOIN game_team_match ON game_summary.game_id = game_team_match.game_id
JOIN team_info as home_team ON home_team.team_id = game_team_match.home_team_id
JOIN team_info as vis_team ON vis_team.team_id = game_team_match.visitor_team_id;

ALTER TABLE new_team_in_game
ADD COLUMN team_id int;

UPDATE new_team_in_game
SET team_id = game_with_cities.home_team_id
FROM game_with_cities
WHERE game_with_cities.summary_id=new_team_in_game.summary_id
AND new_team_in_game.home = 'true';

UPDATE new_team_in_game
SET team_id = game_with_cities.vis_team_id
FROM game_with_cities
WHERE game_with_cities.summary_id=new_team_in_game.summary_id
AND new_team_in_game.home = 'false';

DROP TABLE team_in_game;

ALTER TABLE new_team_in_game RENAME TO team_in_game;

-- modify table next_info

CREATE TABLE new_next_info AS

SELECT 
	summary_id,
    next_home_opponent AS opponent,
    next_home_stadium AS stadium,
    next_home_weekday AS weekday,
    TRUE AS home
FROM next_info

UNION ALL

SELECT 
	summary_id,
    next_vis_opponent AS opponent,
    next_vis_stadium AS stadium,
    next_vis_weekday AS weekday,
    FALSE AS home
FROM next_info;

ALTER TABLE new_next_info 
ADD COLUMN team_id int;

UPDATE new_next_info
SET team_id = game_with_cities.home_team_id
FROM game_with_cities
WHERE game_with_cities.summary_id=new_next_info.summary_id
AND new_next_info.home = 'true';

UPDATE new_next_info
SET team_id = game_with_cities.vis_team_id
FROM game_with_cities
WHERE game_with_cities.summary_id=new_next_info.summary_id
AND new_next_info.home = 'false';

DROP TABLE next_info;

ALTER TABLE new_next_info RENAME TO next_info;
ALTER TABLE next_info DROP COLUMN home;

DROP TABLE game_with_cities;

ALTER TABLE player_award DROP COLUMN tm;

-- DROP COLUMN
ALTER TABLE player_info
DROP COLUMN career_ast,
DROP COLUMN career_fg,
DROP COLUMN career_fg3,
DROP COLUMN career_ft,
DROP COLUMN career_g,
DROP COLUMN career_per,
DROP COLUMN career_pts,
DROP COLUMN career_trb,
DROP COLUMN career_ws,
DROP COLUMN career_efg;

-- ALTER TABLE game_team_match
-- DROP COLUMN game_date_est;

-- modify foreign key 
ALTER TABLE person_stats
DROP CONSTRAINT person_stats_summary_id_fk;

ALTER TABLE team_stats
DROP CONSTRAINT team_stats_summary_id_fk;

ALTER TABLE game_info
DROP CONSTRAINT game_info_summary_id_fk;

ALTER TABLE game_summary
DROP CONSTRAINT game_summary_summary_id_fk;

ALTER TABLE person_stats
ADD CONSTRAINT person_stats_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES game_summary(summary_id);

ALTER TABLE team_stats
ADD CONSTRAINT team_stats_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES game_summary(summary_id);

ALTER TABLE team_in_game
ADD CONSTRAINT team_in_game_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES game_summary(summary_id);

ALTER TABLE next_info
ADD CONSTRAINT next_info_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES game_summary(summary_id);

ALTER TABLE game_info
ADD CONSTRAINT game_info_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES game_summary(summary_id);

ALTER TABLE next_info
ADD CONSTRAINT next_info_pk PRIMARY KEY (summary_id, team_id);

ALTER TABLE team_in_game
ADD CONSTRAINT team_in_game_pk PRIMARY KEY (summary_id, team_id);

ALTER TABLE champion_history
RENAME COLUMN mvp_nationality TO nationality_of_mvp_player;

ALTER TABLE champion_history
RENAME COLUMN eastern_champion_team_id TO eastern_champion;

ALTER TABLE champion_history
RENAME COLUMN western_champion_team_id TO western_champion;

ALTER TABLE champion_history
RENAME COLUMN nba_champion_team_id TO nba_champion;

ALTER TABLE champion_history
RENAME COLUMN nba_vice_champion_team_id TO nba_vice_champion;

ALTER TABLE champion_history
RENAME COLUMN mvp_player_id TO mvp_player;

ALTER TABLE champion_history
DROP COLUMN mvp_status;

ALTER TABLE champion_history
DROP COLUMN result;

ALTER TABLE next_info
RENAME COLUMN opponent TO next_game_opponent;

ALTER TABLE next_info
RENAME COLUMN stadium TO next_game_stadium;

ALTER TABLE next_info
RENAME COLUMN weekday TO next_game_weekday;

ALTER TABLE draft_combine_stats
DROP COLUMN wingspan_ft_in,
DROP COLUMN standing_reach_ft_in;

ALTER TABLE draft_combine_stats
RENAME COLUMN body_fat_pct TO percentage_of_body_fat;

ALTER TABLE draft_combine_stats
DROP COLUMN modified_lane_agility_time;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_fifteen_corner_left TO fifteen_corner_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_fifteen_break_left TO fifteen_break_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_fifteen_top_key TO fifteen_top_key;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_fifteen_break_right TO fifteen_break_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_fifteen_corner_right TO fifteen_corner_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_college_corner_left TO college_corner_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_college_break_left TO college_break_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_college_top_key TO college_top_key;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_college_break_right TO college_break_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_college_corner_right TO college_corner_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_nba_corner_left TO nba_corner_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_nba_break_left TO nba_break_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_nba_top_key TO nba_top_key;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_nba_break_right TO nba_break_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN spot_nba_corner_right TO nba_corner_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_fifteen_break_left TO off_dribble_fifteen_break_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_fifteen_top_key TO off_dribble_fifteen_top_key;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_fifteen_break_right TO off_dribble_fifteen_break_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_college_break_left TO off_dribble_college_break_left;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_college_top_key TO off_dribble_college_top_key;

ALTER TABLE draft_combine_stats
RENAME COLUMN off_drib_college_break_right TO off_dribble_college_break_right;

ALTER TABLE draft_combine_stats
RENAME COLUMN bench_press TO number_of_bench_press;

ALTER TABLE person_stats
RENAME COLUMN ast TO number_of_assist;

ALTER TABLE person_stats
RENAME COLUMN blk TO number_of_block;

ALTER TABLE person_stats
RENAME COLUMN dreb TO number_of_defensive_rebounds;

ALTER TABLE person_stats
RENAME COLUMN fg3a TO number_of_three_point_field_goals_attempted;

ALTER TABLE person_stats
RENAME COLUMN fg3m TO number_of_three_point_field_goals_made;

ALTER TABLE person_stats
RENAME COLUMN fg3_pct TO percentage_of_three_point_field_goal_made;

ALTER TABLE person_stats
RENAME COLUMN fga TO number_of_field_goals_attempted;

ALTER TABLE person_stats
RENAME COLUMN fgm TO number_of_field_goals_made;

ALTER TABLE person_stats
RENAME COLUMN fg_pct TO percentage_of_field_goal_made;

ALTER TABLE person_stats
RENAME COLUMN fta TO number_of_free_throws_attempted;

ALTER TABLE person_stats
RENAME COLUMN ftm TO number_of_free_throws_made;

ALTER TABLE person_stats
RENAME COLUMN ft_pct TO percentage_of_free_throw_made;

ALTER TABLE person_stats
RENAME COLUMN min TO minutes_played;

ALTER TABLE person_stats
RENAME COLUMN oreb TO number_of_offensive_rebounds;

ALTER TABLE person_stats
RENAME COLUMN pf TO number_of_personal_fouls;

ALTER TABLE person_stats
RENAME COLUMN pts TO number_of_points;

ALTER TABLE person_stats
RENAME COLUMN reb TO number_of_rebound;

ALTER TABLE person_stats
RENAME COLUMN stl TO number_of_steal;

ALTER TABLE person_stats
RENAME COLUMN turnover TO number_of_turnover;

ALTER TABLE player_award
DROP COLUMN winner,
DROP COLUMN first,
DROP COLUMN share,
DROP COLUMN age,
DROP COLUMN pts_won,
DROP COLUMN pts_max;

ALTER TABLE team_stats
RENAME COLUMN team_ast TO team_assist;

ALTER TABLE team_stats
RENAME COLUMN team_fg3_pct TO team_percentage_of_three_point_field_goal_made;

ALTER TABLE team_stats
RENAME COLUMN team_fg_pct TO team_percentage_of_field_goal_made;

ALTER TABLE team_stats
RENAME COLUMN team_losses TO number_of_losses_in_season;

ALTER TABLE team_stats
RENAME COLUMN team_pts TO team_points;

ALTER TABLE team_stats
RENAME COLUMN team_pts_qtr1 TO team_points_in_quarter1;

ALTER TABLE team_stats
RENAME COLUMN team_pts_qtr2 TO team_points_in_quarter2;

ALTER TABLE team_stats
RENAME COLUMN team_pts_qtr3 TO team_points_in_quarter3;

ALTER TABLE team_stats
RENAME COLUMN team_pts_qtr4 TO team_points_in_quarter4;

ALTER TABLE team_stats
RENAME COLUMN team_reb TO team_rebound;

ALTER TABLE team_stats
RENAME COLUMN team_tov TO team_turnover;

ALTER TABLE team_stats
RENAME COLUMN team_wins TO number_of_wins_in_season;

ALTER TABLE top_player_career
DROP COLUMN first,
DROP COLUMN share,
DROP COLUMN ws,
DROP COLUMN ws_48,
DROP COLUMN age;

ALTER TABLE top_player_career
RENAME COLUMN g TO games_played;

ALTER TABLE top_player_career
RENAME COLUMN mp TO minutes_played_per_game;

ALTER TABLE top_player_career
RENAME COLUMN pts TO points_per_game;

ALTER TABLE top_player_career
RENAME COLUMN trb TO total_rebounds_per_game;

ALTER TABLE top_player_career
RENAME COLUMN ast TO assists_per_game;

ALTER TABLE top_player_career
RENAME COLUMN stl TO steals_per_game;

ALTER TABLE top_player_career
RENAME COLUMN blk TO blocks_per_game;

ALTER TABLE top_player_career
RENAME COLUMN fg_perc TO field_goal_percentage;

ALTER TABLE top_player_career
RENAME COLUMN threep_perc TO percentage_of_three_point_field_goal;

ALTER TABLE top_player_career
RENAME COLUMN ft_perc TO free_throw_percentage;

ALTER TABLE next_info
RENAME COLUMN next_game_opponent TO opponent_of_next_game;

ALTER TABLE next_info
RENAME COLUMN next_game_stadium TO stadium_of_next_game;

ALTER TABLE next_info
RENAME COLUMN next_game_weekday TO weekday_of_next_game;

ALTER TABLE team_stats
DROP COLUMN number_of_losses_in_season,
DROP COLUMN number_of_wins_in_season;

ALTER TABLE team_info
DROP COLUMN min_year;

ALTER TABLE team_info
DROP COLUMN city;

-- ALTER TABLE team_info
-- DROP COLUMN nickname;

ALTER TABLE team_info
DROP COLUMN abbreviation;

ALTER TABLE team_info
RENAME COLUMN max_year TO disbandment_year;

ALTER TABLE team_info
RENAME COLUMN yearfounded TO founded_year;

ALTER TABLE team_info
RENAME COLUMN arenacapacity TO arena_capacity;

ALTER TABLE player_salary
DROP COLUMN season_end;

ALTER TABLE player_salary
DROP COLUMN season_start;

ALTER TABLE player_award RENAME TO nba_player_award;
ALTER TABLE person_stats RENAME TO player_game_stats;
ALTER TABLE team_stats RENAME TO team_game_stats;
ALTER TABLE champion_history RENAME TO nba_champion_history;
ALTER TABLE next_info RENAME TO next_game_information;
ALTER TABLE draft_combine_stats RENAME TO nba_draft_combine_stats;
ALTER TABLE team_info RENAME TO nba_team_information;
ALTER TABLE player_info RENAME TO nba_player_information;
ALTER TABLE game_info RENAME TO game_information;
ALTER TABLE player_salary RENAME TO nba_player_affiliation;
ALTER TABLE game_team_match RENAME TO nba_game_home_away_record;
DROP TABLE top_player_career;