# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 DesignWeaver3D
# SPDX-FileNotice: Part of the Detessellate addon.

import freecad.Detessellate as module
from importlib.resources import as_file, files

resources = files(module) / 'Resources'
icons = resources / 'Icons'

def asIcon(name: str):
    file = name + '.svg'
    icon = icons / file
    with as_file(icon) as path:
        return str(path)
