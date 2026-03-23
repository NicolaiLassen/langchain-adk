from __future__ import annotations

from orxhestra.skills.skill import Skill
from orxhestra.skills.skill_store import BaseSkillStore


class InMemorySkillStore(BaseSkillStore):
    """Dict-backed skill store for local dev and tests."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._by_id: dict[str, Skill] = {}
        self._by_name: dict[str, Skill] = {}
        for skill in (skills or []):
            self.add(skill)

    def add(self, skill: Skill) -> None:
        """Register a skill.

        Parameters
        ----------
        skill : Skill
            The skill to register.

        Raises
        ------
        ValueError
            If a skill with the same name is already registered.
        """
        if skill.name in self._by_name:
            raise ValueError(f"Skill '{skill.name}' already registered.")
        self._by_id[skill.id] = skill
        self._by_name[skill.name] = skill

    async def get_skill(self, skill_id: str) -> Skill | None:
        """Retrieve a skill by its unique ID.

        Parameters
        ----------
        skill_id : str
            The unique skill identifier.

        Returns
        -------
        Skill or None
            The matching skill, or None if not found.
        """
        return self._by_id.get(skill_id)

    async def get_by_name(self, name: str) -> Skill | None:
        """Retrieve a skill by its name.

        Parameters
        ----------
        name : str
            The skill name to look up.

        Returns
        -------
        Skill or None
            The matching skill, or None if not found.
        """
        return self._by_name.get(name)

    async def list_skills(self) -> list[Skill]:
        """Return all registered skills.

        Returns
        -------
        list[Skill]
            All skills in registration order.
        """
        return list(self._by_name.values())
