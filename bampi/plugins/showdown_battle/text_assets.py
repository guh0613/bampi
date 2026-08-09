STATUS_TEXT = {
    "brn": "灼伤",
    "par": "麻痹",
    "slp": "睡眠",
    "tox": "剧毒",
    "psn": "中毒",
    "frz": "冰冻",
    "drowsy": "瞌睡",
    "fnt": "倒下",
}

STAT_TEXT = {
    "atk": "攻击",
    "def": "防御",
    "spa": "特攻",
    "spd": "特防",
    "spe": "速度",
    "accuracy": "命中",
    "evasion": "闪避",
}

WEATHER_TEXT = {
    "RainDance": "下雨",
    "SunnyDay": "大晴天",
    "Sandstorm": "沙暴",
    "Hail": "冰雹",
    "Snow": "大雪",
    "HarshSunshine": "大日照",
    "HeavyRain": "大雨",
    "StrongWinds": "乱流",
}

MOVE_CATEGORY_TEXT = {
    "Physical": "物理",
    "Special": "特殊",
    "Status": "变化",
}

MOVE_TARGET_TEXT = {
    "normal": "对方单体",
    "self": "自身",
    "any": "任意目标",
    "adjacentAlly": "相邻我方",
    "adjacentAllyOrSelf": "相邻我方或自身",
    "adjacentFoe": "相邻对方",
    "adjacentFoes": "相邻对方全部",
    "allAdjacentFoes": "相邻对方全部",
    "allAdjacent": "相邻所有宝可梦",
    "allies": "我方全体",
    "allyPokemon": "我方在场宝可梦",
    "allySide": "我方场地",
    "allyTeam": "我方队伍",
    "foeSide": "对方场地",
    "foeTeam": "对方队伍",
    "opposingSide": "对方场地",
    "randomNormal": "随机对方",
    "all": "全场所有",
    "allPokemon": "全场所有",
}

FIELD_TEXT = {
    "Misty Terrain": "薄雾场地",
    "Grassy Terrain": "青草场地",
    "Electric Terrain": "电气场地",
    "Psychic Terrain": "精神场地",
    "Wonder Room": "奇迹空间",
    "Trick Room": "戏法空间",
    "Magic Room": "魔法空间",
    "Gravity": "重力",
    "Trick Room Lapse": "戏法空间消失",
}

SIDE_CONDITION_TEXT = {
    "Reflect": "反射壁",
    "Light Screen": "光墙",
    "Aurora Veil": "极光幕",
    "Tailwind": "顺风",
    "Safeguard": "神秘守护",
    "Mist": "白雾",
    "Stealth Rock": "隐形岩",
    "Spikes": "撒菱",
    "Toxic Spikes": "毒菱",
    "Sticky Web": "粘网",
}

TERRAIN_EFFECTS = {
    "Misty Terrain",
    "Grassy Terrain",
    "Electric Terrain",
    "Psychic Terrain",
}

VOLATILE_TEXT = {
    "Substitute": "替身",
    "Dynamax": "极巨化",
}

# Raw [from] effect tokens that are not covered by any entity catalog.
EFFECT_CAUSE_TEXT = {
    "recoil": "反作用力",
    "confusion": "混乱",
    "drain": "吸取效果",
}

BOOST_ORDER = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

PRIVATE_CHECK_HINT = "提示：发送“check <编号>”查看当前招式详情（双打可用“check2 <编号>”查看第2位），或发送“check 招式名”查询任意招式。"
PRIVATE_STATUS_HINT = "提示：发送“战况”查看对战状态；双打行动可用 move1/move2 组合，如“move1 1; move2 2”。也可使用“check <编号>”了解招式详情。"
