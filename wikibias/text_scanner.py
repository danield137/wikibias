from typing import Callable
from .schemas import BiasFinding
from .llm import extract_json_from_result, create_agent


def analyze_loaded_language(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect emotionally or politically charged 'loaded' words.

    Args:
        full_text: The full paragraph text for context
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='loaded_language'
    """
    agent = create_agent(
        name="LoadedLanguageAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are an expert at detecting loaded language in text. 
        Analyze the following text. Find any 'loaded language' (emotionally or politically charged words).
        
        Look for:
        1. Terms that carry implicit judgment or bias (e.g., "colonization" vs "immigration/settlement", "occupied" vs "disputed")
        2. Language that implies illegitimacy or delegitimization 
        3. Terms that frame one side negatively while ignoring context
        4. Words that imply ethnic cleansing or genocidal intent without evidence
        5. Selective use of terminology that favors one narrative

        For each, return a BiasFinding object with kind 'loaded_language', the text span, offset, 
        a strength score, and a neutral alternative in the explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "loaded_language",
              "strength": <0.0-1.0>,
              "text": "exact text span, \"double quotes\" escaped",
              "offset": [start_index, end_index],
              "explanation": "Why this is loaded and neutral alternative"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse loaded_language analysis: {str(e)[:100]}")
        return []


def analyze_asymmetric_labeling(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect asymmetric labeling of opposing groups.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='asymmetric_labeling'
    """
    agent = create_agent(
        name="AsymmetricLabelingAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text for asymmetric labeling of opposing groups.
        
        IMPORTANT: Apply a HIGH THRESHOLD for bias detection. Only flag CLEAR and NON-DEBATABLE distortions.
        
        DO NOT flag:
        - Factual reporting of specific events where one party is the aggressor (e.g., "Group X attacked Group Y")
        - Accurate descriptions of roles in a specific incident (e.g., attackers vs. victims in a documented event)
        - Statements that reflect established facts or widely accepted characterizations
        
        DO flag:
        - Clear distortions that misrepresent reality (e.g., "Hitler was a victim")
        - Systematic use of loaded terms for one group but neutral terms for another group doing similar actions
        - Obvious propaganda language that reverses well-documented aggressor-victim dynamics
        
        Return a BiasFinding object with kind 'asymmetric_labeling', the full text span of 
        the comparison, offset, a strength score (use 0.7+ for clear bias), and an explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "asymmetric_labeling",
              "strength": <0.0-1.0>,
              "text": "exact text span showing comparison",
              "offset": [start_index, end_index],
              "explanation": "Explanation of the clear, non-debatable asymmetry"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse asymmetric_labeling analysis: {str(e)[:100]}")
        return []


def analyze_framing_voice(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect passive voice where the actor is omitted.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='passive_voice_omitted_actor'
    """
    agent = create_agent(
        name="FramingVoiceAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text. Find all instances of passive voice where the actor is omitted
        (e.g., "the villages were bombed" without saying who bombed them).
        
        For each, return a BiasFinding object with kind 'passive_voice_omitted_actor', 
        the text span of the passive phrase, offset, a strength score, and an explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "passive_voice_omitted_actor",
              "strength": <0.0-1.0>,
              "text": "exact passive phrase",
              "offset": [start_index, end_index],
              "explanation": "Explanation of omitted actor"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse framing_voice analysis: {str(e)[:100]}")
        return []


def analyze_statistical_aggregation(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect misleading statistics (aggregation or missing denominator).

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='statistical_aggregation' or 'statistical_missing_denominator'
    """
    agent = create_agent(
        name="StatisticalAggregationAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the statistics in the following text. Find instances of 'statistical_aggregation' 
        (e.g., lumping civilians/combatants) or 'statistical_missing_denominator' 
        (e.g., raw numbers without per-capita context).
        
        For each, return a BiasFinding object with the kind, text span, offset, strength, and explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "statistical_aggregation" or "statistical_missing_denominator",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of the statistical issue"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse statistical_aggregation analysis: {str(e)[:100]}")
        return []


def analyze_omitted_context(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect rhetorical omissions like 'women and children'.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='omitted_context'
    """
    agent = create_agent(
        name="OmittedContextAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text. Find any instances of 'omitted_context' where a common phrase 
        (like 'women and children') is used to rhetorically omit a key group.
        
        For each, return a BiasFinding object with kind 'omitted_context', the text span, offset, 
        strength, and an explanation of what is omitted.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "omitted_context",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of what is omitted"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse omitted_context analysis: {str(e)[:100]}")
        return []


def analyze_certainty_and_hedging(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect claims stated with inappropriate certainty.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='hedging_misuse'
    """
    agent = create_agent(
        name="CertaintyHedgingAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text. Find any 'hedging_misuse' where a disputed claim is stated 
        as a hard fact (lacks hedging) or a known fact is needlessly hedged.
        
        For each, return a BiasFinding object with the kind, text span, offset, strength, and explanation.

        
        ALWAYS Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "hedging_misuse",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of hedging issue"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse certainty_hedging analysis: {str(e)[:100]}")
        return []


def analyze_temporal_framing(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect temporal bias (asymmetric comparisons or superlatives).

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='temporal_framing_asymmetric' or 'temporal_framing_superlative'
    """
    agent = create_agent(
        name="TemporalFramingAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text for temporal bias. Find 'temporal_framing_asymmetric' 
        (mismatched time comparisons like 'this week' vs 'all of 2005') or 
        'temporal_framing_superlative' (e.g., 'worst since...').
        
        For each, return a BiasFinding object with the kind, text span, offset, strength, and explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "temporal_framing_asymmetric" or "temporal_framing_superlative",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of temporal bias"
            }}
          ]
        }}
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse temporal_framing analysis: {str(e)[:100]}")
        return []


def analyze_emphasis_bias(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect emphasis bias via adjectives/adverbs.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='emphasis_bias_minimizer' or 'emphasis_bias_maximizer'
    """
    agent = create_agent(
        name="EmphasisBiasAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text. Find any 'emphasis_bias' words (minimizers like 'only', 'merely' 
        or maximizers like 'staggering', 'clearly').
        
        For each, return a BiasFinding object with the kind, text span, offset, strength, and explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "emphasis_bias_minimizer" or "emphasis_bias_maximizer",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of emphasis bias"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse emphasis_bias analysis: {str(e)[:100]}")
        return []


def analyze_false_balance(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect false balance (presenting fringe views as equal to consensus).

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='false_balance'
    """
    agent = create_agent(
        name="FalseBalanceAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text. Find any instances of 'false_balance' where a fringe viewpoint 
        is presented as a valid counterpoint to a consensus.
        
        For each, return a BiasFinding object with the kind, text span, offset, strength, and explanation.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "false_balance",
              "strength": <0.0-1.0>,
              "text": "exact text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of false balance"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse false_balance analysis: {str(e)[:100]}")
        return []


def analyze_narrative_framing(full_text: str, article_topic: str, get_model: Callable) -> list[BiasFinding]:
    """Analyze the rhetorical purpose of including a claim (meta-tool).

    Args:
        full_text: The text to analyze
        article_topic: The topic of the article for context
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='narrative_framing'
    """
    agent = create_agent(
        name="NarrativeFramingAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the rhetorical purpose of the following text, given the article's topic. 
        Does the *inclusion* of this text, even if factual, create a 'narrative_framing' bias 
        (e.g., 'victimhood', 'aggression', 'undue_weight')?
        
        If so, return a BiasFinding object for the entire span.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "narrative_framing",
              "strength": <0.0-1.0>,
              "text": "entire text span",
              "offset": [start_index, end_index],
              "explanation": "Explanation of narrative framing"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}\n\nTopic: {article_topic}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse narrative_framing analysis: {str(e)[:100]}")
        return []


def analyze_missing_attribution(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect claims that lack proper attribution or sourcing.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='missing_attribution'
    """
    agent = create_agent(
        name="MissingAttributionAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the following text for claims that require attribution but lack it.
        
        For each claim, determine:
        1. Is this a well-accepted axiom in science, culture, or common knowledge? (e.g., "water boils at 100°C", "World War II ended in 1945")
        2. Or does this claim require a source/attribution? (e.g., specific motivations, intentions, goals, interpretations, disputed facts)
        
        DO NOT flag:
        - Well-established historical facts
        - Scientific facts and natural laws
        - Common knowledge and widely accepted information
        - Statements with clear citations already present
        
        DO flag:
        - Claims about motivations, intentions, or goals without attribution (e.g., "with the stated goal of..." - stated by whom?)
        - Disputed or controversial claims presented as fact
        - Specific numbers or statistics without sources
        - Interpretations or opinions presented as objective facts
        
        For each finding, return a BiasFinding object with kind 'missing_attribution', 
        the text span, offset, strength score, and explanation of why attribution is needed.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "missing_attribution",
              "strength": <0.0-1.0>,
              "text": "exact text span lacking attribution",
              "offset": [start_index, end_index],
              "explanation": "Why this claim needs attribution and what information is missing"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse missing_attribution analysis: {str(e)[:100]}")
        return []

def analyze_political_alignment(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Analyze the political alignment or ideological framing of claims.

    Args:
        full_text: The text to analyze
        get_model: Function to get the LLM model

    Returns:
        list of BiasFinding objects with kind='political_alignment'
    """
    agent = create_agent(
        name="PoliticalAlignmentAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. Analyze the political alignment or ideological framing of the following text AS A WHOLE.
        
        IMPORTANT: Read the ENTIRE text carefully before making judgments. Consider the overall balance and context.
        
        Use a strength score from -1.0 to +1.0 where:
        - -1.0 = Strong left-leaning / progressive bias
        - -0.5 = Moderate left-leaning bias
        - 0.0 = Neutral / balanced
        - +0.5 = Moderate right-leaning bias
        - +1.0 = Strong right-leaning / conservative bias
        
        Consider the ENTIRE text, not just individual sentences:
        - Word choice and framing across all content
        - What facts are emphasized vs. omitted throughout
        - How events or groups are characterized overall
        - Underlying assumptions or value judgments
        - The balance of perspectives presented
        
        DO NOT flag as biased:
        - Factual reporting that happens to mention one side first
        - Balanced text that presents both perspectives
        - Text that is neutral in tone even if discussing controversial topics
        
        DO flag:
        - Clear, systematic bias in how information is presented
        - Obvious framing that favors one political perspective
        - Significant omission or minimization of key facts that would shift the narrative
        
        Be nuanced: Only flag when there is clear evidence of political bias in the OVERALL text.
        
        Output ONLY valid JSON in this format:
        {{
          "findings": [
            {{
              "kind": "political_alignment",
              "strength": <-1.0 to +1.0>,
              "text": "exact text span showing alignment",
              "offset": [start_index, end_index],
              "explanation": "Explanation of the political alignment detected, considering the full context and overall balance"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse political_alignment analysis: {str(e)[:100]}")
        return []

def analyze_missing_context(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect when important historical or political context is missing."""
    agent = create_agent(
        name="MissingContextAnalyzer", 
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are an expert at identifying missing context that creates bias through omission.
        
        General principles to check:
        1. Historical claims without relevant background (e.g., mentioning a territory without its legal status history)
        2. Actions described without their causes or precipitating events
        3. One group's claims/connections mentioned without acknowledging competing claims
        4. Terms used without defining their contested meanings
        5. Events described without security, economic, or political context that motivated them
        6. Selective timeframes that exclude relevant precedents or consequences
        
        When unclear about bias, use this approach:
        - Imagine a debate between opposing viewpoints on this topic
        - List the top 3 points each side would make about the missing context
        - If one side's arguments feel significantly less represented in the text, that indicates bias
        - Consider: Would adding the missing context change a reader's understanding?
        
        Output ONLY valid JSON:
        {{
          "findings": [
            {{
              "kind": "missing_context",
              "strength": <0.0-1.0>,
              "text": "statement lacking context",
              "offset": [start_index, end_index],
              "explanation": "what context is missing and why it matters for neutral understanding"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse missing_context analysis: {str(e)[:100]}")
        return []


def analyze_historical_revisionism(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Check for historical inaccuracies or misleading claims."""
    agent = create_agent(
        name="HistoricalRevisionismAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are a historical fact-checker. Identify historical revisionism or inaccuracies.
        
        General patterns to identify:
        1. Claims about group intentions without evidence (e.g., "X wanted to eliminate Y")
        2. Implying modern nation-states or concepts existed in different historical contexts
        3. Stating one interpretation of contested history as fact
        4. Ignoring documented legal frameworks or international agreements
        5. Attributing actions to entire groups rather than specific actors/factions
        6. Mischaracterizing mainstream movements by their extremist elements
        7. Selective presentation of facts that distorts overall understanding
        8. Anachronistic judgments (applying modern values to historical events)
        
        When evaluating claims, consider:
        - Would historians from different backgrounds dispute this characterization?
        - Is this the scholarly consensus or a partisan interpretation?
        - Are minority/extremist views being presented as mainstream?
        
        Debate test: If scholars debated this claim:
        - What would each side argue?
        - Is the text presenting only one side's interpretation as fact?
        - Would adding "according to X perspective" make it more accurate?
        
        Output ONLY valid JSON:
        {{
          "findings": [
            {{
              "kind": "historical_revisionism",
              "strength": <0.0-1.0>,
              "text": "historically inaccurate or misleading statement",
              "offset": [start_index, end_index],
              "explanation": "why this is disputed/inaccurate and what the scholarly consensus or debate actually is"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse historical_revisionism analysis: {str(e)[:100]}")
        return []


def analyze_framing_bias(full_text: str, get_model: Callable) -> list[BiasFinding]:
    """Detect biased framing and selective presentation of facts."""
    agent = create_agent(
        name="FramingBiasAnalyzer",
        instructions=f"""
        You are a staff writer in a prestigious newspaper well regarded for its neutrality and fact checking. You are an expert at detecting how facts are selectively framed to create bias.
        
        General framing biases to detect:
        1. Contested claims presented as established facts without qualification
        2. One-sided victim/aggressor narratives in complex conflicts
        3. Selective emphasis on negative actions of one party
        4. Claims about group intentions without evidence or sourcing
        5. Using charged terms for one side's actions, neutral terms for similar actions by another
        6. Active voice for one side's negative actions, passive voice for another's
        7. Legitimizing language for one side, delegitimizing language for another
        8. Omitting that multiple narratives exist on controversial topics
        
        Key test - The Debate Framework:
        When you see a potentially controversial statement:
        1. Imagine advocates for different perspectives debating this issue
        2. What would each side say about this framing?
        3. Does the current framing clearly favor one perspective?
        4. Would a neutral observer need to hear both framings to understand?
        
        Example: "Group X colonized the territory" vs "Group X immigrated to the territory"
        - One frame implies illegitimacy, the other legitimacy
        - A neutral framing might be: "Group X arrived/settled in the territory" or acknowledge the debate
        
        Output ONLY valid JSON:
        {{
          "findings": [
            {{
              "kind": "framing_bias",
              "strength": <0.0-1.0>,
              "text": "biased framing statement",
              "offset": [start_index, end_index],
              "explanation": "how this framing favors one perspective and what a neutral framing would be"
            }}
          ]
        }}

        If for some reason you cannot compute a value for a field, use null.
        """,
        get_model=get_model,
    )

    result = agent.run(f"Text: {full_text}")
    try:
        data = extract_json_from_result(result)
        findings = data.get("findings", [])
        return [BiasFinding(**f) for f in findings]
    except Exception as e:
        print(f"  Warning: Failed to parse framing_bias analysis: {str(e)[:100]}")
        return []

# Add these to your TEXT_SCANNER_TOOLS dictionary:
TEXT_SCANNER_TOOLS = {
    "analyze_loaded_language": analyze_loaded_language,
    "analyze_asymmetric_labeling": analyze_asymmetric_labeling,
    "analyze_framing_voice": analyze_framing_voice,
    "analyze_statistical_aggregation": analyze_statistical_aggregation,
    "analyze_omitted_context": analyze_omitted_context,
    "analyze_certainty_and_hedging": analyze_certainty_and_hedging,
    "analyze_temporal_framing": analyze_temporal_framing,
    "analyze_emphasis_bias": analyze_emphasis_bias,
    "analyze_false_balance": analyze_false_balance,
    "analyze_narrative_framing": analyze_narrative_framing,
    "analyze_missing_attribution": analyze_missing_attribution,
    "analyze_political_alignment": analyze_political_alignment,
    "analyze_missing_context": analyze_missing_context,
    "analyze_historical_revisionism": analyze_historical_revisionism,
    "analyze_framing_bias": analyze_framing_bias,
}
