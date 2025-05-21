
-- This file contains auxiliary functions and other definitions required to
-- make the provenance analysis of the provided examples possible.

-- Installation: psql < aux.sql
-- tested on PostgreSQL Version 14.1


--------------------------------------------------------------------------------
-- Types


-- location id type
DROP DOMAIN IF EXISTS loc_t CASCADE;
CREATE DOMAIN loc_t AS INTEGER;

-- row identifier type (a.k.a. tuple identifier = tuid)
DROP DOMAIN IF EXISTS tuid_t CASCADE;
CREATE DOMAIN tuid_t AS INTEGER;

-- provenance annotation type
DROP DOMAIN IF EXISTS pset_t CASCADE;
CREATE DOMAIN pset_t BIGINT[] DEFAULT '{}';


--------------------------------------------------------------------------------
-- ID Generators


-- tuid generator
DROP SEQUENCE IF EXISTS tuid_seq CASCADE;
CREATE SEQUENCE tuid_seq;

-- annotation generator
DROP SEQUENCE IF EXISTS annot_seq CASCADE;
CREATE SEQUENCE annot_seq;


--------------------------------------------------------------------------------
-- Logging
-- in the paper, this is described as the "interim protocol"
-- table-based logging (see below) is one possible implementation of it


-- *** log0
DROP TABLE IF EXISTS log0 CASCADE;
CREATE TABLE log0 (location loc_t NOT NULL,
                   tuidout tuid_t NOT NULL);
ALTER TABLE log0 ADD PRIMARY KEY (location);
ALTER TABLE log0 ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readlog(v_location loc_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT j.tuidout
      FROM log0 AS j
     WHERE j.location=v_location
$$ LANGUAGE SQL STABLE;

-- read
-- for non-tabular context
CREATE FUNCTION readOne(v_location loc_t)
RETURNS tuid_t AS
$$
    SELECT j.tuidout
      FROM log0 AS j
     WHERE j.location=v_location
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writelog(v_location loc_t)
RETURNS tuid_t AS
$$
DECLARE
    res tuid_t;
BEGIN
    INSERT INTO log0 (location)
        VALUES (v_location)
        RETURNING tuidout INTO res;
        RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.tuidout
                 FROM log0 AS j
                WHERE j.location=v_location);
END;
$$ LANGUAGE PLPGSQL VOLATILE;



-- *** log1
DROP TABLE IF EXISTS log1 CASCADE;
CREATE TABLE log1 (location loc_t NOT NULL,
                   tuidout tuid_t NOT NULL,
                   tuid1 tuid_t NOT NULL);
ALTER TABLE log1 ADD PRIMARY KEY (location, tuid1);
ALTER TABLE log1 ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readlog(v_location loc_t, v_tuid1 tuid_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT j.tuidout
      FROM log1 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
$$ LANGUAGE SQL STABLE;

-- read
-- for non-tabular context
CREATE FUNCTION readOne(v_location loc_t, v_tuid1 tuid_t)
RETURNS tuid_t AS
$$
    SELECT j.tuidout
      FROM log1 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writelog(v_location loc_t, v_tuid1 tuid_t)
RETURNS tuid_t AS
$$
DECLARE
    res tuid_t;
BEGIN
    INSERT INTO log1 (location, tuid1)
        VALUES (v_location, v_tuid1)
        RETURNING tuidout INTO res;
        RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.tuidout
                 FROM log1 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1);
END;
$$ LANGUAGE PLPGSQL VOLATILE;



-- *** log2
DROP TABLE IF EXISTS log2 CASCADE;
CREATE TABLE log2 (location loc_t NOT NULL,
                   tuidout tuid_t NOT NULL,
                   tuid1 tuid_t NOT NULL,
                   tuid2 tuid_t NOT NULL);
ALTER TABLE log2 ADD PRIMARY KEY (location, tuid1, tuid2);
ALTER TABLE log2 ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readlog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT j.tuidout
      FROM log2 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
       AND j.tuid2=v_tuid2
$$ LANGUAGE SQL STABLE;

-- read
-- for non-tabular contexts
CREATE FUNCTION readOne(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t)
RETURNS tuid_t AS
$$
    SELECT j.tuidout
      FROM log2 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
       AND j.tuid2=v_tuid2
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writelog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t)
RETURNS tuid_t AS
$$
DECLARE
    res tuid_t;
BEGIN
    INSERT INTO log2 (location, tuid1, tuid2)
        VALUES (v_location, v_tuid1, v_tuid2)
        RETURNING tuidout INTO res;
        RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.tuidout
                 FROM log2 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1
                  AND j.tuid2=v_tuid2);
END;
$$ LANGUAGE PLPGSQL VOLATILE;



-- *** log3
DROP TABLE IF EXISTS log3 CASCADE;
CREATE TABLE log3 (location loc_t NOT NULL,
                   tuidout tuid_t NOT NULL,
                   tuid1 tuid_t NOT NULL,
                   tuid2 tuid_t NOT NULL,
                   tuid3 tuid_t NOT NULL);
ALTER TABLE log3 ADD PRIMARY KEY (location, tuid1, tuid2, tuid3);
ALTER TABLE log3 ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readlog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t, v_tuid3 tuid_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT j.tuidout
      FROM log3 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
       AND j.tuid2=v_tuid2
       AND j.tuid3=v_tuid3
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writelog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t, v_tuid3 tuid_t)
RETURNS tuid_t AS
$$
DECLARE
    res tuid_t;
BEGIN
    INSERT INTO log3 (location, tuid1, tuid2, tuid3)
        VALUES (v_location, v_tuid1, v_tuid2, v_tuid3)
        RETURNING tuidout INTO res;
        RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.tuidout
                 FROM log3 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1
                  AND j.tuid3=v_tuid3
                  AND j.tuid2=v_tuid2);
END;
$$ LANGUAGE PLPGSQL VOLATILE;



-- *** case1: with 1 tuid
DROP TABLE IF EXISTS logcase1 CASCADE;
CREATE TABLE logcase1 (location loc_t NOT NULL,
                       branchid integer NOT NULL,
                       tuid1 tuid_t NOT NULL);
ALTER TABLE logcase1 ADD PRIMARY KEY (location, tuid1);
ALTER TABLE logcase1 ALTER COLUMN branchid SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readcase(v_location loc_t, v_tuid1 tuid_t)
RETURNS integer AS
$$
    SELECT j.branchid
      FROM logcase1 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writecase(v_location loc_t, v_tuid1 tuid_t, v_branchid integer)
RETURNS integer AS
$$
DECLARE
    res integer;
BEGIN
    INSERT INTO logcase1 (location, branchid, tuid1)
        VALUES (v_location, v_branchid, v_tuid1)
        RETURNING v_branchid INTO res;
    RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.branchid
                 FROM logcase1 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1);
END;
$$ LANGUAGE PLPGSQL VOLATILE;


-- *** case2: with 2 tuids
DROP TABLE IF EXISTS logcase2 CASCADE;
CREATE TABLE logcase2 (location loc_t NOT NULL,
                       branchid integer NOT NULL,
                       tuid1 tuid_t NOT NULL,
                       tuid2 tuid_t NOT NULL);
ALTER TABLE logcase2 ADD PRIMARY KEY (location, tuid1, tuid2);
ALTER TABLE logcase2 ALTER COLUMN branchid SET DEFAULT NEXTVAL('tuid_seq');

-- read
CREATE FUNCTION readcase(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t)
RETURNS integer AS
$$
    SELECT j.branchid
      FROM logcase2 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
       AND j.tuid2=v_tuid2
$$ LANGUAGE SQL STABLE;

-- write
CREATE FUNCTION writecase(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t, v_branchid integer)
RETURNS integer AS
$$
DECLARE
    res integer;
BEGIN
    INSERT INTO logcase2 (location, branchid, tuid1, tuid2)
        VALUES (v_location, v_branchid, v_tuid1, v_tuid2)
        RETURNING v_branchid INTO res;
    RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.branchid
                 FROM logcase2 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1
                  AND j.tuid2=v_tuid2);
END;
$$ LANGUAGE PLPGSQL VOLATILE;


-- *** log4
DROP TABLE IF EXISTS log4 CASCADE;
CREATE TABLE log4 (location loc_t NOT NULL,
                   tuidout tuid_t NOT NULL,
                   tuid1 tuid_t NOT NULL,
                   tuid2 tuid_t NOT NULL,
                   tuid3 tuid_t NOT NULL,
                   tuid4 tuid_t NOT NULL);
ALTER TABLE log4 ADD PRIMARY KEY (location, tuid1, tuid2, tuid3, tuid4);
ALTER TABLE log4 ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

-- read
DROP FUNCTION IF EXISTS readlog(loc_t, tuid_t, tuid_t, tuid_t, tuid_t) CASCADE;
CREATE FUNCTION readlog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t, v_tuid3 tuid_t, v_tuid4 tuid_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT j.tuidout
      FROM log4 AS j
     WHERE j.location=v_location
       AND j.tuid1=v_tuid1
       AND j.tuid2=v_tuid2
       AND j.tuid3=v_tuid3
       AND j.tuid4=v_tuid4
$$ LANGUAGE SQL STABLE;

-- write
DROP FUNCTION IF EXISTS writelog(loc_t, tuid_t, tuid_t, tuid_t, tuid_t) CASCADE;
CREATE FUNCTION writelog(v_location loc_t, v_tuid1 tuid_t, v_tuid2 tuid_t, v_tuid3 tuid_t, v_tuid4 tuid_t)
RETURNS tuid_t AS
$$
DECLARE
    res tuid_t;
BEGIN
    INSERT INTO log4 (location, tuid1, tuid2, tuid3, tuid4)
        VALUES (v_location, v_tuid1, v_tuid2, v_tuid3, v_tuid4)
        RETURNING tuidout INTO res;
        RETURN res;
EXCEPTION
    WHEN UNIQUE_VIOLATION THEN
        -- readLog() got inlined
        -- advantage: gain better performance through less context switches
        RETURN(SELECT j.tuidout
                 FROM log4 AS j
                WHERE j.location=v_location
                  AND j.tuid1=v_tuid1
                  AND j.tuid2=v_tuid2
                  AND j.tuid3=v_tuid3
                  AND j.tuid4=v_tuid4);
END;
$$ LANGUAGE PLPGSQL VOLATILE;

--- *** loggrp
DROP TABLE IF EXISTS loggrp CASCADE;
CREATE TABLE loggrp (
    location loc_t NOT NULL,
    tuidout tuid_t NOT NULL,
    tuidset tuid_t NOT NULL
);
ALTER TABLE loggrp ADD PRIMARY KEY (location, tuidout, tuidset);
ALTER TABLE loggrp ALTER COLUMN tuidout SET DEFAULT NEXTVAL('tuid_seq');

DROP FUNCTION IF EXISTS writegrp(loc_t, tuid_t[]) CASCADE;
CREATE FUNCTION writegrp(v_location loc_t, v_tuidset tuid_t[])
RETURNS tuid_t AS
$$
DECLARE
    group_id tuid_t;
    tid tuid_t;
BEGIN
    -- 새로운 그룹 ID 생성
    SELECT nextval('tuid_seq') INTO group_id;

    -- tuidset의 각 요소를 row로 삽입
    FOREACH tid IN ARRAY v_tuidset
    LOOP
        INSERT INTO loggrp(location, tuidout, tuidset)
        VALUES (v_location, group_id, tid);
    END LOOP;

    RETURN group_id;
END;
$$ LANGUAGE PLPGSQL VOLATILE;

DROP FUNCTION IF EXISTS readgrp(loc_t, tuid_t) CASCADE;
CREATE FUNCTION readgrp(v_location loc_t, v_groupid tuid_t)
RETURNS TABLE(tuid tuid_t) AS
$$
    SELECT g.tuidset
    FROM loggrp AS g
    WHERE g.location = v_location AND g.tuidout = v_groupid
$$ LANGUAGE SQL STABLE;

--------------------------------------------------------------------------------
-- Management of Provenance Sets

-- sets are represented as PG arrays


-- turn into Why-provenance
DROP FUNCTION IF EXISTS toY(a pset_t) CASCADE;
CREATE FUNCTION toY(a pset_t)
RETURNS pset_t AS
$$
-- make Why-provenance negative and do early duplicate removal
    SELECT ARRAY(SELECT DISTINCT -abs(x.unnested)
                   FROM unnest(a) AS x(unnested))::pset_t
$$ LANGUAGE SQL;


-- drop duplicates
DROP FUNCTION IF EXISTS dd(a pset_t) CASCADE;
CREATE FUNCTION dd(a pset_t)
    RETURNS pset_t AS
$$
    SELECT ARRAY(SELECT DISTINCT v.unnested
                   FROM unnest(a) AS v(unnested))::pset_t
$$ LANGUAGE SQL;


-- shortcut for an empty provenance set
CREATE FUNCTION empty()
RETURNS pset_t AS $$
    SELECT ARRAY[]::pset_t
$$ LANGUAGE SQL;



