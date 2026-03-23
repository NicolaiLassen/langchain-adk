"""Skill tools - list_skills and load_skill as LangChain tools.

"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from orxhestra.skills.skill_store import BaseSkillStore


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""
    name: str = Field(description="The name of the skill to load.")


def make_load_skill_tool(store: BaseSkillStore) -> BaseTool:
    """Create a load_skill tool bound to the given skill store.

    The LLM calls this tool to fetch skill instructions by name and
    incorporate them into its working context.

    Parameters
    ----------
    store : BaseSkillStore
        The skill store to load skills from.

    Returns
    -------
    BaseTool
        A LangChain BaseTool.
    """
    async def load_skill(name: str) -> str:
        """Load a skill's full instruction content by name."""
        skill = await store.get_by_name(name)
        if skill is None:
            available = [s.name for s in await store.list_skills()]
            return (
                f"Skill '{name}' not found. "
                f"Available skills: {', '.join(available) or 'none'}."
            )
        return f"# {skill.name}\n\n{skill.content}"

    return StructuredTool.from_function(
        coroutine=load_skill,
        name="load_skill",
        description=(
            "Load a skill's instructions by name. "
            "Use this when a skill is relevant to the current task."
        ),
        args_schema=LoadSkillInput,
    )


def make_list_skills_tool(store: BaseSkillStore) -> BaseTool:
    """Create a list_skills tool that returns the skill roster.

    Parameters
    ----------
    store : BaseSkillStore
        The skill store to list skills from.

    Returns
    -------
    BaseTool
        A LangChain BaseTool.
    """
    async def list_skills() -> str:
        """List all available skills with their names and descriptions."""
        skills = await store.list_skills()
        if not skills:
            return "No skills available."
        lines = ["Available skills:"] + [
            f"  - {s.name}: {s.description}" for s in skills
        ]
        return "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=list_skills,
        name="list_skills",
        description="List all available skills with their names and descriptions.",
    )
