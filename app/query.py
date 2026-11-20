"""Parse football queries into conditions the local index can actually verify."""
import re
import unicodedata
from dataclasses import dataclass, field


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for short, long in [("didn't", "did not"), ("doesn't", "does not"), ("wasn't", "was not")]:
        text = text.replace(short, long)
    return " ".join(re.findall(r"[a-z0-9]+", text.replace("'s", "")))


# Each recognized phrase contributes a condition. Unknown words are retained.
VOCAB = {
    "piece": {"corner": "corner|corners|corner kick|corner kicks", "free_kick": "free kick|free kicks",
              "penalty": "penalty|penalties|penalty kick", "set_piece": "set piece|set pieces"},
    "pattern": {"counter_attack": "counterattack|counterattacks|counter attack|counter attacks|on the break",
                "switch_of_play": "switch of play|switches of play|switch play|switch the play"},
    "step": {"regain": "regain possession|regains possession|regained possession|regaining possession|regain|regains|regained|regaining|win it back|wins it back|won it back|win the ball back|wins the ball back|winning the ball back|won the ball back|win back possession|wins back possession|win possession|winning possession|winning it back|ball recovery|turnover",
             "progression": "progression|advance|advances|advancing|progress|progresses|progressing|progress the ball|progresses the ball|carry forward|move forward|moves forward",
             "receive": "receive|receives|receiving|received|reception|first touch|controls",
             "pass": "pass|passes|passing|passed", "cross": "cross|crosses|crossing|crossed",
             "shot": "shot|shots|shoot|shoots|shooting|strike|effort|attempt on goal",
             "header": "header|headers|headed|heading", "dribble": "dribble|dribbles|dribbling|carry|carries",
             "tackle": "tackle|tackles|tackling", "interception": "interception|intercept|intercepts|interceptions",
             "clearance": "clearance|clearances|clears|cleared",
             "save": "save|saves|goalkeeper save", "foul": "foul|fouls|fouled", "throw_in": "throw in|throw ins",
             "goal_kick": "goal kick|goal kicks"},
    "result": {"saved": "saved|saved shot|shot saved|shot was saved|shot is saved",
               "goal": "goal|goals|scores|scored|scoring|finish in the net",
               "blocked": "blocked|blocked shot", "on_target": "on target|on goal",
               "off_target": "off target|misses|missed|goes wide",
               "intercepted": "intercepted"},
    "shape": {"half_turn": "half turn|half turned", "open": "open body|facing forward|facing goal",
              "back_to_goal": "back to goal|back to the goal|with his back"},
    "length": {"short": "short", "medium": "medium", "long": "long"},
    "pressure": {"tight": "under tight pressure|under heavy pressure|tightly marked|tightly pressed",
                 "any": "under pressure|being pressed", "none": "unpressured|without pressure|no pressure"},
    "style": {"diving": "diving|diving goalkeeper|goalkeeper diving"},
    "origin": {"left": "from the left wing|from the left flank|from the left", "right": "from the right wing|from the right flank|from the right"},
    "destination": {"opposite_box": "far side of the penalty area|far side of the box",
                    "box": "penalty area|penalty box|the box", "left": "to the left wing", "right": "to the right wing"},
    "relation": {"neg": "did not|does not|without|never|not|no", "or": "or",
                 "then": "then|followed by|leading to|led to|lead to|leads to|before",
                 "terminal": "result in|resulted in|results in|resulting in|end in|ending in|ended in|ends in",
                 "after": "after|following|from a|from an|from", "open_play": "open play"},
    "role": {"taker": "taken by|delivered by|takes|took", "scorer": "finished by|scored by|finished off by",
             "shooter": "shot by", "keeper": "saved by"},
}
TEAM_NAMES = {
    "ARS": "ars|arsenal", "MU": "mu|man utd|man united|manchester utd|manchester united|united",
    "MC": "mc|mci|man city|manchester city|city",
}
FULL_NAMES = {"Mainoo": "Kobbie Mainoo", "Fernandes": "Bruno Fernandes", "Saka": "Bukayo Saka",
              "Merino": "Mikel Merino"}
FILLER = set("a an the show me find all any every where when which who that this give of for it his her their ball player players team clips clip times moments moment live play by in to on with and from into towards as was is are resulting resulted ends ended leading led lead".split())


@dataclass
class Query:
    tags: list[tuple[str, str]]
    unknown: list[str]
    problem: str = ""
    role_players: dict = field(default_factory=dict)
    step_players: dict = field(default_factory=dict)

    def values(self, kind):
        return [v for k, v in self.tags if k == kind]

    def one(self, kind):
        return next((v for k, v in self.tags if k == kind), None)


def vocabulary(teams, players):
    phrases = {}
    for kind, values in VOCAB.items():
        for value, aliases in values.items():
            for term in aliases.split("|"):
                phrases[term] = (kind, value)
    for abbreviation, aliases in TEAM_NAMES.items():
        team = next((name for name in teams if normalize(name) in aliases.split("|")), abbreviation)
        for term in aliases.split("|"):
            phrases[term] = ("team", team)
    for name in teams:
        phrases[normalize(name)] = ("team", name)
    for name in players:
        phrases[normalize(name)] = ("player", name)
        if name in FULL_NAMES:
            phrases[normalize(FULL_NAMES[name])] = ("player", name)
    return phrases


def parse(text, phrases):
    text = normalize(text)
    # In active voice the keeper is mentioned first, but the attempt happens first.
    teams = "|".join(re.escape(term) for term, (kind, _) in phrases.items() if kind == "team")
    text = re.sub(rf"\b(?:(?:goalkeeper|keeper) )?saves? (?:a |an |the )?(?=(?:(?:{teams}) )?(?:header|shot)s?\b)",
                  "save after ", text)
    text = re.sub(r"\bsaved by (?:the |a )?(?:keeper|goalkeeper)\b", "followed by save", text)
    text = re.sub(r"\bsaved headers?\b", "header followed by save", text)
    words = text.split()
    tags, unknown = [], []
    i = 0
    while i < len(words):
        for end in range(min(i + 8, len(words)), i, -1):
            tag = phrases.get(" ".join(words[i:end]))
            if tag:
                # "Shot by" expresses an action as well as the following person's role.
                # Keep both so before/after ordering sees the action.
                event = {("role", "shooter"): ("step", "shot"), ("role", "keeper"): ("step", "save"),
                         ("role", "scorer"): ("result", "goal")}.get(tag)
                if event and tags[-1:] != [event]:
                    tags.append(event)
                tags.append(tag)
                i = end
                break
        else:
            if words[i] not in FILLER:
                unknown.append(words[i])
            i += 1
    q = Query(tags, unknown)
    relations = q.values("relation")
    if "after" in relations and sum(r in {"after", "then", "terminal"} for r in relations) > 1:
        q.problem = "Multiple nested time conditions need separate verification."
    if relations.count("neg") > 1:
        q.problem = "The scope of multiple exclusions could not be verified."
    if "or" in q.values("relation"):
        q.problem = "Alternative conditions are not supported."
    # A team attached to the save in a passage is the defending side.
    # Keep it distinct from the team that took the corner or won possession.
    if q.one("piece") or q.one("pattern") or "regain" in q.values("step") or len(q.values("step")) > 1:
        for i, (kind, team) in enumerate(tags):
            if kind == "team" and (tags[i+1:i+2] == [("step", "save")] or
                                   (i and tags[i-1] in {("role", "keeper"), ("step", "save")})):
                tags[i] = ("saving_team", team)
    if len(set(q.values("team"))) > 1 or len(set(q.values("saving_team"))) > 1:
        q.problem = "The roles of the two teams could not be verified."
    for kind in ("shape", "length", "pressure", "origin", "destination", "piece", "pattern", "result"):
        if len(set(q.values(kind))) > 1:
            q.problem = "The index cannot verify that combination of conditions."
    for i, (kind, player) in enumerate(tags):
        if kind != "player":
            continue
        previous = tags[i-1] if i else None
        following = tags[i+1] if i+1 < len(tags) else None
        if previous and previous[0] == "role":
            role = previous[1]
        elif following and following[0] == "role":
            role = following[1]
        elif ("result", "goal") in (previous, following):
            role = "scorer"
        elif ("step", "shot") in (previous, following):
            role = "shooter"
        elif ("step", "save") in (previous, following):
            role = "keeper"
        elif any(tag and tag[0] == "step" for tag in (previous, following)):
            step = next(tag[1] for tag in (previous, following) if tag and tag[0] == "step")
            if step in q.step_players and q.step_players[step] != player:
                q.problem = "The named players could not be assigned to distinct actions."
            q.step_players[step] = player
            continue
        elif q.one("piece"):
            role = "taker"
        else:
            role = "actor"
        if role in q.role_players and q.role_players[role] != player:
            q.problem = "The roles of the named players could not be verified."
        q.role_players[role] = player
    for i, (kind, role) in enumerate(tags):
        if kind == "role" and role not in q.role_players:
            nearby = tags[max(0, i-1):i] + tags[i+1:i+2]
            if not any(k in {"team", "saving_team"} for k, _ in nearby):
                q.problem = "A named participant is missing for the requested role."
    return q
