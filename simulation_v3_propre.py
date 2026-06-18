"""Compatibilite: ce fichier delegate vers le point d'entree principal main.py."""

from main import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
