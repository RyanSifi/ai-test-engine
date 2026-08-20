--
-- PostgreSQL database dump
--

-- Dumped from database version 15.4 (Debian 15.4-2.pgdg120+1)
-- Dumped by pg_dump version 15.4 (Debian 15.4-2.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: project_code_context; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.project_code_context (
    id integer NOT NULL,
    project_id character varying(100),
    chunk_type character varying(50),
    file_path character varying(500),
    class_name character varying(255),
    content text,
    embedding public.vector(384)
);


--
-- Name: project_code_context_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.project_code_context_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: project_code_context_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.project_code_context_id_seq OWNED BY public.project_code_context.id;


--
-- Name: project_code_context id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_code_context ALTER COLUMN id SET DEFAULT nextval('public.project_code_context_id_seq'::regclass);


--
-- Name: project_code_context project_code_context_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.project_code_context
    ADD CONSTRAINT project_code_context_pkey PRIMARY KEY (id);


--
-- Name: project_code_context_embedding_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX project_code_context_embedding_idx ON public.project_code_context USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: project_code_context_project_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX project_code_context_project_id_idx ON public.project_code_context USING btree (project_id);


--
-- PostgreSQL database dump complete
--

