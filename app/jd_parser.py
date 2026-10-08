"""Parse a job description into structured data using Claude (Haiku)."""

from typing import Annotated, List, Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

load_dotenv()

MODEL = "claude-haiku-5-5"

TOOL_NAME = "extract_job_description"

NOT_STATED = "not stated"

SYSTEM_PROMPT = f"""You are given the raw text of a job description (JD).

Extract the requested fields using only information explicitly stated in the
JD. Do not infer, guess, or fill in fields from general knowledge about the
company or role.

For any text field that is not mentioned in the JD, use the literal string
"{NOT_STATED}". For any list field with no items mentioned in the JD, use an
empty list. Never leave a field out or invent a value.

Skill extraction rules (required_skills, nice_to_have_skills):
- List short, normalized skill names only (e.g. "Python", "REST APIs",
  "Prompt Engineering") - never copy full requirement sentences or phrases.
- If the JD states a combined phrase covering more than one skill (e.g.
  "Python and AI/ML concepts" or "APIs, databases, and software development
  concepts"), split it into separate, atomic skill entries.
- required_skills comes only from explicit requirement/qualification
  statements (e.g. a "Requirements"/"Qualifications" section or "must have"
  language). Never pull a skill into required_skills just because it was
  mentioned in a responsibilities/duties section.
- Any skill described with hedging language - "an advantage", "preferred",
  "nice to have", "familiarity with", "bonus", "plus" - goes in
  nice_to_have_skills, even if that same skill is also mentioned elsewhere
  (e.g. in responsibilities). Hedged language always wins over a mention
  elsewhere.
- If a skill (not hedged) appears in both a required/must-have section and
  a preferred/nice-to-have section, list it only in required_skills.
- After drafting required_skills and nice_to_have_skills, de-duplicate
  across BOTH lists together as one combined set - not each list in
  isolation. If a broader skill you listed covers a narrower/more generic
  one (e.g. "AI/ML Concepts" covers "Machine Learning" and "Artificial
  Intelligence"; "Cloud Concepts" covers "AWS" only if the JD never names
  AWS specifically), keep only the broader term, in whichever list it
  belongs, and drop the narrower term entirely - even if the narrower term
  would otherwise have landed in the other list. Never end up with both
  the broad and the narrow version anywhere in the output.
- Soft skills (e.g. communication, problem-solving, teamwork, leadership)
  never go in required_skills or nice_to_have_skills - put them in
  soft_skills instead.
- Format every skill name (required_skills, nice_to_have_skills,
  soft_skills) in Title Case (e.g. "Problem Solving", "Machine Learning"),
  except acronyms/initialisms, which keep their standard casing (e.g.
  "REST APIs", "AI/ML", "LLMs", "API").

soft_skills rule: only include a soft skill if the JD explicitly names it
(e.g. the words "communication", "problem solving", "analytical skills",
"teamwork" actually appear). Never infer a soft skill from a responsibility
or duty that merely implies it.

seniority rule: must be exactly one of "fresher", "junior", "mid", "senior",
or "{NOT_STATED}". Use "fresher" when the JD says "fresher" or states
0-1 years of experience. Map other phrasing to the closest of these levels;
use "{NOT_STATED}" only if the JD gives no seniority signal at all.

domain rule: use only the industry or business area the JD explicitly
states (e.g. "fintech", "healthcare", "retail"). Do not infer it from the
job title, the tech stack, or the company name - a company building AI
tools is not automatically in the "AI" domain unless the JD says so. Use
"{NOT_STATED}" if the JD doesn't state an industry/business area.

Call the {TOOL_NAME} tool with the extracted data."""


SkillName = Annotated[str, StringConstraints(strip_whitespace=True, max_length=40)]


class JobDescription(BaseModel):
    """Structured fields extracted from a job description."""

    model_config = ConfigDict(extra="forbid")

    job_title: str = Field(description='The job title, or "not stated" if missing.')
    company: str = Field(description='The hiring company name, or "not stated" if missing.')
    location: str = Field(description='The job location, or "not stated" if missing.')
    work_mode: Literal["remote", "hybrid", "on-site", "not stated"] = Field(
        description='Whether the role is remote, hybrid, on-site, or "not stated" if missing.'
    )
    seniority: Literal["fresher", "junior", "mid", "senior", "not stated"] = Field(
        description='Seniority level. "fresher" if the JD says fresher or 0-1 years of '
        'experience. "not stated" only if the JD gives no seniority signal at all.'
    )
    years_experience: str = Field(
        description='Years of experience required, as stated in the JD (e.g. "3-5 years"), '
        'or "not stated" if missing.'
    )
    required_skills: List[SkillName] = Field(
        description='Required/must-have technical skills, from explicit requirement '
        'statements only (not from responsibilities), as short Title Case skill names '
        '(e.g. "Python", "REST APIs", "Prompt Engineering") - never full sentences, never '
        "hedged/preferred skills, never soft skills, never near-duplicates of a broader "
        "skill already listed. Empty list if none are stated."
    )
    nice_to_have_skills: List[SkillName] = Field(
        description='Preferred/nice-to-have technical skills, same short Title Case rules '
        "as required_skills. Includes any skill described with hedging language "
        '("an advantage", "preferred", "nice to have", "familiarity with", "bonus", "plus") '
        "even if that same skill is also mentioned in responsibilities. Excludes any skill "
        "already listed in required_skills and any near-duplicate of a broader skill already "
        "listed. Empty list if none are stated."
    )
    soft_skills: List[SkillName] = Field(
        description='Soft/interpersonal skills as short Title Case names '
        '(e.g. "Communication", "Problem Solving", "Teamwork") - never technical skills, and '
        "only skills the JD explicitly names, never inferred from a responsibility. Empty "
        "list if none are stated."
    )
    domain: str = Field(
        description='The industry or business area the role/company operates in '
        '(e.g. "fintech", "healthcare"). Only use what the JD explicitly states - '
        'do not infer it from the job title, tech stack, or company name. '
        '"not stated" if the JD doesn\'t say.'
    )
    key_responsibilities: List[str] = Field(
        description="List of key responsibilities/duties. Empty list if none are stated."
    )
    education: str = Field(
        description='Education requirement, or "not stated" if missing.'
    )


def _build_tool() -> dict:
    schema = JobDescription.model_json_schema()
    schema.pop("title", None)
    return {
        "name": TOOL_NAME,
        "description": "Record the structured fields extracted from a job description.",
        "strict": True,
        "input_schema": schema,
    }


def parse_jd(jd_text: str) -> JobDescription:
    """Send a job description to Claude and return structured data.

    Raises if Claude does not return a valid tool call matching the schema.
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[_build_tool()],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": jd_text}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return JobDescription.model_validate(tool_use.input)
