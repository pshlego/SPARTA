ALTER TABLE player_info RENAME COLUMN _id to player_id;

ALTER TABLE player_info
  ADD COLUMN height_in INTEGER;

UPDATE player_info
SET height_in =
  (CAST(split_part(height, '-', 1) AS INT) * 12)
  + CAST(split_part(height, '-', 2) AS INT);

ALTER TABLE player_info
  DROP COLUMN height;

ALTER TABLE player_info
  RENAME COLUMN height_in TO height;
  
-- add team_name to team_info
ALTER TABLE team_info
ADD COLUMN team_name VARCHAR(100);

UPDATE team_totals
set abbreviation='PHX'
WHERE abbreviation = 'PHO';

UPDATE team_totals
set abbreviation='BKN'
WHERE abbreviation = 'BRK';

UPDATE team_info
SET team_name = team_totals.team
from team_totals
WHERE team_totals.abbreviation = team_info.abbreviation;

-- add team_id to player_salary
ALTER TABLE player_salary
ADD COLUMN team_id INT;

UPDATE player_salary
set team_id= team_info.team_id
from team_info
WHERE player_salary.team = team_info.team_name;

UPDATE player_salary
SET team_id = 1610612751
WHERE team = 'New Jersey Nets';

UPDATE player_salary
SET team_id = 1610612766
WHERE team = 'Charlotte Hornets';

UPDATE player_salary
SET team_id = 1610612740
WHERE team = 'New Orleans/Oklahoma City Hornets';

UPDATE player_salary
SET team_id = 1610612740
WHERE team = 'New Orleans Hornets';

UPDATE player_salary
SET team_id = 1610612763
WHERE team = 'Vancouver Grizzlies';

UPDATE player_salary
SET team_id = 1610612764
WHERE team = 'Washington Bullets';

UPDATE player_salary
SET team_id = 1610612760
WHERE team = 'Seattle SuperSonics';

UPDATE player_salary
SET team_id = 1610612758
WHERE team = 'Kansas City Kings';

-- Player salary에 team 이름 없는 거  4개 존재

DELETE FROM player_info WHERE draft_year LIKE '1st%' OR draft_year LIKE '2nd%' OR draft_year LIKE '8th%';

ALTER TABLE player_info
ALTER COLUMN draft_year TYPE INTEGER USING draft_year::INTEGER;

-- DELETE 302 (3293->2991)
DELETE from player_award
where season > 2018;

DELETE FROM player_award
WHERE winner = 'False';

ALTER TABLE player_info
ADD COLUMN birthdate_temp DATE;

UPDATE player_info
SET birthdate_temp = TO_DATE(birthdate, 'Month DD, YYYY');

ALTER TABLE player_info ADD COLUMN birthyear INTEGER;

UPDATE player_info
SET birthyear = EXTRACT(YEAR FROM birthdate_temp);

ALTER TABLE player_info DROP COLUMN birthdate_temp;

ALTER TABLE player_award
ADD COLUMN birthyear INT;

UPDATE player_award
SET birthyear = season - age;

-- 생년이 1년 차이나고 이름이 같은 경우가 없음 -> name, birthyear로 player_id 부여

ALTER TABLE player_award
DROP COLUMN seas_id,
DROP COLUMN player_id;

ALTER TABLE player_award
ADD COLUMN player_id varchar(100);

UPDATE player_award
SET player_id = player_info.player_id
from player_info
where player_award.player = player_info.name
and (player_award.birthyear=player_info.birthyear or player_award.birthyear = player_info.birthyear+1);

-- top_player_career 연결
-- delete 39 (474->435)
DELETE from top_player_career
where year > 2018;

ALTER TABLE top_player_career
ADD COLUMN birthyear INT;

UPDATE top_player_career
SET birthyear = year - age;

UPDATE top_player_career
SET player = 'Peja Stojakovic'
where player = 'Peja Stojakovi?';
UPDATE top_player_career
SET player = 'Manu Ginobili'
where player = 'Manu Gin�bili';
UPDATE top_player_career
SET player = 'Goran Dragic'
where player = 'Goran Dragi?';

ALTER TABLE top_player_career
ADD COLUMN player_id varchar(100);

UPDATE top_player_career
SET player_id = player_info.player_id
from player_info
where top_player_career.player = player_info.name
and (top_player_career.birthyear=player_info.birthyear or top_player_career.birthyear = player_info.birthyear+1);

-- mvp info 연결
ALTER TABLE mvp_info RENAME TO champion_history;

ALTER TABLE champion_history
ADD COLUMN name VARCHAR(100);

UPDATE champion_history
SET name = player_award.player
from player_award
WHERE player_award.season = champion_history.year 
and player_award.award = 'nba mvp' and player_award.winner = 'True';

ALTER TABLE champion_history DROP COLUMN mvp_name;
ALTER TABLE champion_history RENAME COLUMN name TO mvp_name;

-- player_info에 이름으로 중복된 게 없으므로 바로 player_id 부여
ALTER TABLE champion_history
ADD COLUMN mvp_player_id varchar(100);

UPDATE champion_history
SET mvp_player_id = player_info.player_id
from player_info
where champion_history.mvp_name = player_info.name;

-- add team id to mvp info
ALTER TABLE champion_history
ADD COLUMN western_champion_team_id INT;

UPDATE champion_history
SET western_champion_team_id=player_salary.team_id
FROM player_salary
WHERE champion_history.western_champion=player_salary.team;

ALTER TABLE champion_history
ADD COLUMN eastern_champion_team_id INT;

UPDATE champion_history
SET eastern_champion_team_id=player_salary.team_id
FROM player_salary
WHERE champion_history.eastern_champion=player_salary.team;

ALTER TABLE champion_history
ADD COLUMN nba_champion_team_id INT;

UPDATE champion_history
SET nba_champion_team_id=player_salary.team_id
FROM player_salary
WHERE champion_history.nba_champion=player_salary.team;

ALTER TABLE champion_history
ADD COLUMN nba_vice_champion_team_id INT;

UPDATE champion_history
SET nba_vice_champion_team_id=player_salary.team_id
FROM player_salary
WHERE champion_history.nba_vice_champion=player_salary.team;

-- draft_combine_stats
--DELETE 279 (1476->1197)
DELETE FROM draft_combine_stats
where season > 2018;

DELETE FROM draft_combine_stats
where player_id = 1628992;

ALTER TABLE draft_combine_stats
ADD COLUMN person_id VARCHAR(100);

UPDATE draft_combine_stats
SET person_id = 'brownde03'
WHERE player_id = 200793;

UPDATE draft_combine_stats
SET person_id = player_info.player_id
from player_info
where draft_combine_stats.player_name = player_info.name and draft_combine_stats.season = player_info.draft_year
and draft_combine_stats.person_id is null;

UPDATE draft_combine_stats
SET person_id = player_info.player_id
from player_info
where draft_combine_stats.player_name = player_info.name
and draft_combine_stats.person_id is null;

ALTER TABLE draft_combine_stats
DROP COLUMN player_id;

ALTER TABLE draft_combine_stats
RENAME COLUMN person_id TO player_id;

DELETE from draft_combine_stats
where player_id is null;

-- drop unnecessary columns
ALTER TABLE player_salary
DROP COLUMN league;

ALTER TABLE champion_history
DROP COLUMN index,
DROP COLUMN western_champion,
DROP COLUMN eastern_champion,
DROP COLUMN nba_champion,
DROP COLUMN nba_vice_champion,
DROP COLUMN mvp_name,
DROP COLUMN mvp_height_m,
DROP COLUMN mvp_height_ft,
DROP COLUMN mvp_position,
DROP COLUMN mvp_team;

ALTER TABLE player_award
DROP COLUMN player,
DROP COLUMN birthyear;

ALTER TABLE top_player_career
DROP COLUMN player,
DROP COLUMN tm,
DROP COLUMN birthyear;

ALTER TABLE draft_combine_stats
DROP COLUMN first_name,
DROP COLUMN last_name,
DROP COLUMN player_name,
DROP COLUMN position,
DROP COLUMN height_wo_shoes,
DROP COLUMN height_wo_shoes_ft_in,
DROP COLUMN height_w_shoes,
DROP COLUMN height_w_shoes_ft_in,
DROP COLUMN weight;

-- connect to text db

-- 1. game join
CREATE TABLE game_from_rotowire AS
SELECT games.id, games.date, code
from games
join team_in_games on games.id = team_in_games.game_id
join cache_team_names on team_in_games.id = cache_team_names.team_in_game_id;

ALTER TABLE game_team_match
ALTER COLUMN game_date_est TYPE date USING to_date(game_date_est, 'YYYY-MM-DD');

ALTER TABLE game_team_match
DROP COLUMN team_id_home,
DROP COLUMN pts_home,
DROP COLUMN fg_pct_home,
DROP COLUMN ft_pct_home,
DROP COLUMN fg3_pct_home,
DROP COLUMN ast_home,
DROP COLUMN reb_home,
DROP COLUMN team_id_away,
DROP COLUMN pts_away,
DROP COLUMN fg_pct_away,
DROP COLUMN ft_pct_away,
DROP COLUMN fg3_pct_away,
DROP COLUMN ast_away,
DROP COLUMN reb_away,
DROP COLUMN home_team_wins;

WITH cte AS (
    SELECT *,
           ROW_NUMBER() OVER(PARTITION BY game_date_est, game_id, game_status_text, home_team_id, visitor_team_id, season 
                             ORDER BY game_date_est) AS rn
    FROM game_team_match
)

DELETE FROM game_team_match
USING cte
WHERE game_team_match.game_date_est = cte.game_date_est
  AND game_team_match.game_id = cte.game_id
  AND game_team_match.game_status_text = cte.game_status_text
  AND game_team_match.home_team_id = cte.home_team_id
  AND game_team_match.visitor_team_id = cte.visitor_team_id
  AND game_team_match.season = cte.season
  AND cte.rn > 1;

ALTER TABLE game_team_match
ADD COLUMN game_new_id int;

ALTER TABLE game_team_match
ADD COLUMN home_team VARCHAR(10),
ADD COLUMN visit_team VARCHAR(10);

UPDATE game_team_match
SET home_team = team_info.abbreviation
FROM team_info
where team_info.team_id = game_team_match.home_team_id;

UPDATE game_team_match
SET visit_team = team_info.abbreviation
FROM team_info
where team_info.team_id = game_team_match.visitor_team_id;

UPDATE game_team_match
SET game_new_id = game_from_rotowire.id
from game_from_rotowire
where game_team_match.game_date_est=game_from_rotowire.date
and (game_team_match.home_team = game_from_rotowire.code or game_team_match.visit_team = game_from_rotowire.code);

ALTER TABLE game_team_match
DROP COLUMN game_status_text,
DROP COLUMN game_id,
DROP COLUMN home_team,
DROP COLUMN visit_team,
DROP COLUMN season;

ALTER TABLE game_team_match
RENAME COLUMN game_new_id to game_id;

-- 2. player Join
ALTER TABLE prep_person_stats
ADD COLUMN other_name varchar(100);

UPDATE prep_person_stats
SET other_name = name_inconsist.other_name
FROM name_inconsist
WHERE name_inconsist.PLAYER_NAME = prep_person_stats."PLAYER_NAME";

ALTER TABLE prep_person_stats
ADD COLUMN player_id varchar(100);

-- 동명이인 먼저 처리하기
UPDATE prep_person_stats
SET player_id = 'tayloje03'
WHERE "PLAYER_NAME"='Jeff Taylor';
UPDATE prep_person_stats
SET player_id = 'willire02'
WHERE "PLAYER_NAME"='Reggie Williams';
UPDATE prep_person_stats
SET player_id = 'leeda02'
WHERE "PLAYER_NAME"='David Lee';
UPDATE prep_person_stats
SET player_id = 'jamesmi02'
WHERE "PLAYER_NAME"='Mike James';
UPDATE prep_person_stats
SET player_id = 'dunlemi02'
WHERE "PLAYER_NAME"='Mike Dunleavy';
UPDATE prep_person_stats
SET player_id = 'hendege02'
WHERE "PLAYER_NAME"='Gerald Henderson';
UPDATE prep_person_stats
SET player_id = 'johnsch04'
WHERE "PLAYER_NAME"='Chris Johnson';

UPDATE prep_person_stats
SET player_id = player_info.player_id
FROM player_info
WHERE prep_person_stats.player_id is null 
and (player_info.name = prep_person_stats."PLAYER_NAME" or player_info.name = prep_person_stats.other_name);

ALTER TABLE prep_person_stats DROP COLUMN other_name;

-- 3. team join
ALTER TABLE prep_team_stats
ADD COLUMN abbreviation VARCHAR(10);

UPDATE prep_team_stats
SET abbreviation = team_names.code
from team_names
where prep_team_stats."TEAM-NAME" = team_names.name;

ALTER TABLE prep_team_stats
ADD COLUMN team_id int;

UPDATE prep_team_stats
SET team_id = team_info.team_id
from team_info
where prep_team_stats.abbreviation = team_info.abbreviation;

UPDATE prep_team_stats
SET team_id = 1610612751
WHERE abbreviation = 'NJN';

UPDATE prep_team_stats
SET team_id = 1610612740
WHERE abbreviation = 'NOH';

ALTER TABLE prep_team_stats DROP COLUMN abbreviation;

DROP TABLE discrepancy_adjustments CASCADE;
DROP TABLE discrepancies_person_in_team_in_game_periods CASCADE;
DROP TABLE league_structures CASCADE;
DROP TABLE discrepancies_discrepancy_adjustments CASCADE;
DROP TABLE discrepancies CASCADE;
DROP TABLE discrepancies_person_in_team_in_games CASCADE;
DROP TABLE discrepancies_play_statistic_statistics CASCADE;
DROP TABLE discrepancies_statistics CASCADE;
DROP TABLE discrepancies_team_in_games CASCADE;
DROP TABLE division_structures CASCADE;
DROP TABLE divisions CASCADE;
DROP TABLE game_periods CASCADE;
DROP TABLE month_names CASCADE;
DROP TABLE games_stadia CASCADE;
DROP TABLE dataset_splits CASCADE;
DROP TABLE people_teams CASCADE;
DROP TABLE person_in_team_in_game_periods CASCADE;
DROP TABLE person_in_team_in_games CASCADE;
DROP TABLE people CASCADE;
DROP TABLE person_in_team_in_game_periods_play_statistics CASCADE;
DROP TABLE person_in_team_in_game_periods_statistics CASCADE;
DROP TABLE person_in_team_in_seasons CASCADE;
DROP TABLE person_in_team_in_games_positions CASCADE;
DROP TABLE person_in_team_in_games_statistics CASCADE;
DROP TABLE person_in_team_in_games_team_in_games CASCADE;
DROP TABLE place_types CASCADE;
DROP TABLE plays CASCADE;
DROP TABLE playoff_series CASCADE;
DROP TABLE playoff_series_team_in_seasons CASCADE;
DROP TABLE places_in_places CASCADE;
DROP TABLE play_statistics CASCADE;
DROP TABLE positions CASCADE;
DROP TABLE places_team_in_seasons CASCADE;
DROP TABLE places_teams CASCADE;
DROP TABLE play_statistic_reasons CASCADE;
DROP TABLE play_statistic_reason_types CASCADE;
DROP TABLE play_statistics_team_in_game_periods CASCADE;
DROP TABLE playoffs CASCADE;
DROP TABLE playoff_round_types CASCADE;
DROP TABLE plays_team_in_games CASCADE;
DROP TABLE playoff_series_numbers CASCADE;
DROP TABLE seasons CASCADE;
DROP TABLE schema_migrations CASCADE;
DROP TABLE roles CASCADE;
DROP TABLE sports CASCADE;
DROP TABLE stadia CASCADE;
DROP TABLE stadia_team_in_seasons CASCADE;
DROP TABLE stadia_teams CASCADE;
DROP TABLE statistic_types CASCADE;
DROP TABLE team_names_teams CASCADE;
DROP TABLE teams CASCADE;
DROP TABLE team_in_seasons CASCADE;
DROP TABLE team_in_playoff_games CASCADE;
DROP TABLE team_in_seasons_team_names CASCADE;
DROP TABLE team_names CASCADE;
DROP TABLE leagues CASCADE;
DROP TABLE conferences CASCADE;
DROP TABLE statistics CASCADE;
DROP TABLE team_in_games CASCADE;
DROP TABLE people_positions CASCADE;
DROP TABLE team_in_game_periods CASCADE;
DROP TABLE places CASCADE;
DROP TABLE playoff_rounds CASCADE;
DROP TABLE statistics_team_in_game_periods CASCADE;
DROP TABLE statistics_team_in_games CASCADE;
DROP TABLE cache_game_names CASCADE;
DROP TABLE cache_game_period_names CASCADE;
DROP TABLE cache_person_in_team_in_game_periods_all_statistics CASCADE;
DROP TABLE cache_person_in_team_in_games_all_statistics CASCADE;
DROP TABLE cache_person_names CASCADE;
DROP TABLE cache_team_in_game_periods_all_statistics CASCADE;
DROP TABLE cache_team_in_games_all_statistics CASCADE;
DROP TABLE cache_team_names CASCADE;
DROP TABLE team_totals CASCADE;
DROP TABLE name_inconsist CASCADE;

ALTER TABLE prep_person_stats RENAME TO person_stats;
ALTER TABLE prep_team_stats RENAME TO team_stats;
ALTER TABLE prep_game_information RENAME TO game_info;
ALTER TABLE prep_team_info RENAME TO team_in_game;
ALTER TABLE prep_next_info RENAME TO next_info;

DROP TABLE game_from_rotowire;
DROP TABLE games CASCADE;

ALTER TABLE games_rotowire_entries RENAME TO game_summary;
ALTER TABLE game_summary RENAME COLUMN rotowire_entry_id TO summary_id;

ALTER TABLE rotowire_entries RENAME TO summary;

ALTER TABLE summary
DROP COLUMN dataset_split_id,
DROP COLUMN rw_line,
DROP COLUMN created_at,
DROP COLUMN updated_at;

ALTER TABLE summary RENAME COLUMN id TO summary_id;

ALTER TABLE game_summary DROP CONSTRAINT games_rotowire_entries_rotowire_entry_id_fkey;
ALTER TABLE summary DROP CONSTRAINT rotowire_entries_pkey;

ALTER TABLE team_info DROP COLUMN league_id;

DELETE FROM player_info WHERE player_id IS NULL;
DELETE FROM player_award WHERE player_id IS NULL;

-- add primary key
ALTER TABLE summary
ADD CONSTRAINT summary_pk PRIMARY KEY (summary_id);

ALTER TABLE player_info
ADD CONSTRAINT player_info_pk PRIMARY KEY (player_id);

ALTER TABLE team_info
ADD CONSTRAINT team_info_pk PRIMARY KEY (team_id);

ALTER TABLE game_summary
ADD CONSTRAINT game_summary_pk PRIMARY KEY (summary_id);

ALTER TABLE draft_combine_stats
ADD CONSTRAINT draft_combine_stats_pk PRIMARY KEY (player_id, season);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_pk PRIMARY KEY (year);

ALTER TABLE player_award
ADD CONSTRAINT player_award_pk PRIMARY KEY (season, award, player_id);

DELETE FROM player_salary
where player_id = 'scolalu01' and season = '2014-15' and team = 'Houston Rockets';

UPDATE player_salary
SET team = 'New York Knicks', team_id = 1610612752
WHERE  team is null and player_id = 'cartwbi01';

DELETE from player_salary
where team is null;

ALTER TABLE player_salary
ADD CONSTRAINT player_salary_pk PRIMARY KEY (season, team_id, salary, player_id);

ALTER TABLE top_player_career
ADD CONSTRAINT top_player_career_pk PRIMARY KEY (player_id, year);

DELETE from game_team_match
where game_id is null;

ALTER TABLE game_team_match
ADD CONSTRAINT game_team_match_pk PRIMARY KEY (game_id);

DELETE from person_stats
where summary_id is null;

ALTER TABLE person_stats
ADD CONSTRAINT person_stats_pk PRIMARY KEY (summary_id, player_id);

DELETE from team_stats
where summary_id is null;

UPDATE team_stats
SET team_id = team_info.team_id
from team_info
where team_stats."TEAM-NAME" = team_info.nickname;

ALTER TABLE team_stats
ADD CONSTRAINT team_stats_pk PRIMARY KEY (summary_id, team_id);

DELETE from game_info
where summary_id is null;

ALTER TABLE game_info
ADD CONSTRAINT game_info_pk PRIMARY KEY (summary_id);

DELETE from next_info
where summary_id is null;

ALTER TABLE next_info
ADD CONSTRAINT next_info_pk PRIMARY KEY (summary_id);

DELETE from team_in_game
where summary_id is null;

ALTER TABLE team_in_game
ADD CONSTRAINT team_in_game_pk PRIMARY KEY (summary_id);


-- add foreign key for player id 
ALTER TABLE draft_combine_stats
ADD CONSTRAINT draft_combine_stats_player_id_fk
FOREIGN KEY (player_id) REFERENCES player_info(player_id);

ALTER TABLE player_award
ADD CONSTRAINT player_award_player_id_fk
FOREIGN KEY (player_id) REFERENCES player_info(player_id);

ALTER TABLE top_player_career
ADD CONSTRAINT top_player_career_player_id_fk
FOREIGN KEY (player_id) REFERENCES player_info(player_id);

ALTER TABLE player_salary
ADD CONSTRAINT player_salary_player_id_fk
FOREIGN KEY (player_id) REFERENCES player_info(player_id);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_player_id_fk
FOREIGN KEY (mvp_player_id) REFERENCES player_info(player_id);

ALTER TABLE person_stats
ADD CONSTRAINT person_stats_player_id_fk
FOREIGN KEY (player_id) REFERENCES player_info(player_id);

-- add foreign key for summary id
ALTER TABLE person_stats
ADD CONSTRAINT person_stats_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

ALTER TABLE team_stats
ADD CONSTRAINT team_stats_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

ALTER TABLE team_in_game
ADD CONSTRAINT team_in_game_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

ALTER TABLE next_info
ADD CONSTRAINT next_info_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

ALTER TABLE game_info
ADD CONSTRAINT game_info_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

ALTER TABLE game_summary
ADD CONSTRAINT game_summary_summary_id_fk
FOREIGN KEY (summary_id) REFERENCES summary(summary_id);

-- add foreign key for game id
ALTER TABLE game_summary
ADD CONSTRAINT game_summary_game_id_fk
FOREIGN KEY (game_id) REFERENCES game_team_match(game_id);

-- add foreign key for team id
ALTER TABLE game_team_match
ADD CONSTRAINT game_team_match_home_team_id_fk
FOREIGN KEY (home_team_id) REFERENCES team_info(team_id);

ALTER TABLE game_team_match
ADD CONSTRAINT game_team_match_visitor_team_id_fk
FOREIGN KEY (visitor_team_id) REFERENCES team_info(team_id);

ALTER TABLE player_salary
ADD CONSTRAINT player_salary_team_id_fk
FOREIGN KEY (team_id) REFERENCES team_info(team_id);

ALTER TABLE team_stats
ADD CONSTRAINT team_stats_team_id_fk
FOREIGN KEY (team_id) REFERENCES team_info(team_id);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_western_champion_team_id_fk
FOREIGN KEY (western_champion_team_id) REFERENCES team_info(team_id);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_eastern_champion_team_id_fk
FOREIGN KEY (eastern_champion_team_id) REFERENCES team_info(team_id);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_nba_champion_team_id_fk
FOREIGN KEY (nba_champion_team_id) REFERENCES team_info(team_id);

ALTER TABLE champion_history
ADD CONSTRAINT champion_history_nba_vice_champion_team_id_fk
FOREIGN KEY (nba_vice_champion_team_id) REFERENCES team_info(team_id);