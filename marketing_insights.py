"""
Marketing insights generator.

Translates aggregated emotion detection data into actionable marketing
recommendations tailored to social media content strategy.
"""

from __future__ import annotations

from typing import Any

from emotion_detector import EMOTIONS


# ---------------------------------------------------------------------------
# Emotion → marketing insight mapping
# ---------------------------------------------------------------------------

# Sentiment polarity: positive (+1), neutral (0), negative (−1)
_EMOTION_SENTIMENT: dict[str, int] = {
    "happy": 1,
    "surprise": 1,
    "neutral": 0,
    "sad": -1,
    "angry": -1,
    "disgust": -1,
    "fear": -1,
}

# Engagement potential (high emotions drive more interactions)
_ENGAGEMENT_POTENTIAL: dict[str, str] = {
    "happy": "high",
    "surprise": "high",
    "angry": "high",
    "fear": "medium",
    "sad": "low",
    "disgust": "low",
    "neutral": "low",
}

# Recommended content actions per dominant emotion
_CONTENT_ACTIONS: dict[str, list[str]] = {
    "happy": [
        "Feature this content in brand highlight reels.",
        "Use happy imagery in ad creatives for maximum resonance.",
        "Encourage user-generated content campaigns to amplify positive sentiment.",
    ],
    "surprise": [
        "Leverage novelty elements – surprise content drives high shareability.",
        "Introduce limited-time offers or unexpected product reveals.",
        "A/B-test 'surprise & delight' messaging to boost engagement.",
    ],
    "neutral": [
        "Add stronger emotional hooks (storytelling, humor, inspiration).",
        "Experiment with vibrant visuals or interactive content formats.",
        "Consider personalization to create more relevant emotional connections.",
    ],
    "sad": [
        "Respond with empathy-led messaging to build trust.",
        "Consider cause-marketing or community-support campaigns.",
        "Monitor sentiment closely to prevent brand association with negative topics.",
    ],
    "angry": [
        "Conduct immediate reputation management review.",
        "Engage in proactive customer service outreach.",
        "Avoid promotional content in contexts associated with high anger.",
    ],
    "disgust": [
        "Audit content for anything that may trigger negative reactions.",
        "Refocus messaging on positive product benefits and values.",
        "Pause associated ad campaigns pending a sentiment review.",
    ],
    "fear": [
        "Avoid fear-based messaging unless authentically aligned with safety campaigns.",
        "Provide reassurance and support-focused content.",
        "Highlight reliability and trustworthiness of the brand.",
    ],
}

# Audience segment insights based on dominant emotion
_AUDIENCE_SEGMENTS: dict[str, str] = {
    "happy": "Satisfied customers / brand advocates",
    "surprise": "Curious explorers / early adopters",
    "neutral": "Passive observers / undecided consumers",
    "sad": "Vulnerable or dissatisfied audience",
    "angry": "Frustrated or disengaged consumers",
    "disgust": "Highly critical or alienated audience",
    "fear": "Anxious or risk-averse audience",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emotion_to_sentiment(emotion: str) -> str:
    """Convert an emotion label to a human-readable sentiment.

    Parameters
    ----------
    emotion:
        One of the labels in :data:`emotion_detector.EMOTIONS`.

    Returns
    -------
    str
        ``"positive"``, ``"neutral"``, or ``"negative"``.
    """
    polarity = _EMOTION_SENTIMENT.get(emotion, 0)
    if polarity > 0:
        return "positive"
    if polarity < 0:
        return "negative"
    return "neutral"


def get_content_recommendations(dominant_emotion: str) -> list[str]:
    """Return a list of content strategy recommendations.

    Parameters
    ----------
    dominant_emotion:
        Dominant emotion from the analysis (e.g. ``"happy"``).

    Returns
    -------
    list[str]
        Actionable content strategy items.
    """
    return _CONTENT_ACTIONS.get(dominant_emotion, _CONTENT_ACTIONS["neutral"])


def generate_insights(
    summary: dict[str, Any],
    brand_name: str = "the brand",
) -> dict[str, Any]:
    """Generate marketing insights from an emotion summary.

    Parameters
    ----------
    summary:
        Output of :func:`emotion_detector.get_emotion_summary` or
        :attr:`social_media_analyzer.BatchAnalysisReport.summary`.
    brand_name:
        Optional brand name used in narrative descriptions.

    Returns
    -------
    dict with keys:
        ``dominant_emotion``, ``sentiment``, ``engagement_potential``,
        ``audience_segment``, ``content_recommendations``,
        ``emotion_breakdown``, ``narrative``.
    """
    dominant = summary.get("overall_dominant_emotion", "neutral")
    avg_emotions: dict[str, float] = summary.get("average_emotions", {})
    total_images: int = summary.get("total_images", 0)
    analyzed_images: int = summary.get("analyzed_images", 0)
    total_faces: int = summary.get("total_faces", 0)

    sentiment = emotion_to_sentiment(dominant)
    engagement = _ENGAGEMENT_POTENTIAL.get(dominant, "low")
    audience = _AUDIENCE_SEGMENTS.get(dominant, "General audience")
    recommendations = get_content_recommendations(dominant)

    # Build a concise narrative paragraph
    face_phrase = (
        f"{total_faces} face(s) detected across {analyzed_images} image(s)"
        if analyzed_images > 0
        else "no images successfully analyzed"
    )
    narrative = (
        f"Analysis of {total_images} social media image(s) ({face_phrase}) "
        f"indicates a predominantly **{dominant}** emotional tone for {brand_name}. "
        f"This reflects a **{sentiment}** overall sentiment with **{engagement}** "
        f"engagement potential. The audience profile suggests: {audience.lower()}."
    )

    return {
        "dominant_emotion": dominant,
        "sentiment": sentiment,
        "engagement_potential": engagement,
        "audience_segment": audience,
        "content_recommendations": recommendations,
        "emotion_breakdown": avg_emotions,
        "narrative": narrative,
    }


def generate_campaign_strategy(
    insights: dict[str, Any],
    campaign_goal: str = "brand awareness",
) -> dict[str, Any]:
    """Generate a social-media campaign strategy from marketing insights.

    Parameters
    ----------
    insights:
        Output of :func:`generate_insights`.
    campaign_goal:
        High-level campaign objective (e.g. ``"lead generation"``,
        ``"brand awareness"``, ``"customer retention"``).

    Returns
    -------
    dict with keys:
        ``campaign_goal``, ``dominant_emotion``, ``sentiment``,
        ``recommended_tone``, ``content_types``, ``kpis``,
        ``risk_level``, ``action_items``.
    """
    dominant = insights.get("dominant_emotion", "neutral")
    sentiment = insights.get("sentiment", "neutral")
    engagement = insights.get("engagement_potential", "low")

    # Tone recommendations
    tone_map: dict[str, str] = {
        "happy": "celebratory and upbeat",
        "surprise": "exciting and revealing",
        "neutral": "informative and engaging",
        "sad": "empathetic and supportive",
        "angry": "calm, professional, and solution-oriented",
        "disgust": "transparent and values-driven",
        "fear": "reassuring and trust-building",
    }
    recommended_tone = tone_map.get(dominant, "informative and engaging")

    # Suitable content types per sentiment
    content_types: dict[str, list[str]] = {
        "positive": ["user testimonials", "product showcases", "behind-the-scenes"],
        "neutral": ["educational posts", "polls", "infographics"],
        "negative": ["FAQ content", "customer support posts", "brand story content"],
    }
    suggested_content = content_types.get(sentiment, content_types["neutral"])

    # KPIs aligned with engagement potential and campaign goal
    kpi_map: dict[str, list[str]] = {
        "high": ["shares", "comments", "saves", "click-through rate"],
        "medium": ["impressions", "reach", "story views"],
        "low": ["post reach", "profile visits", "follower growth"],
    }
    kpis = kpi_map.get(engagement, kpi_map["low"])

    risk_level = (
        "high" if dominant in {"angry", "disgust"}
        else "medium" if dominant in {"sad", "fear"}
        else "low"
    )

    action_items = insights.get("content_recommendations", [])

    return {
        "campaign_goal": campaign_goal,
        "dominant_emotion": dominant,
        "sentiment": sentiment,
        "recommended_tone": recommended_tone,
        "content_types": suggested_content,
        "kpis": kpis,
        "risk_level": risk_level,
        "action_items": action_items,
    }


def format_report(
    insights: dict[str, Any],
    strategy: dict[str, Any] | None = None,
) -> str:
    """Format insights (and optionally strategy) as a human-readable report.

    Parameters
    ----------
    insights:
        Output of :func:`generate_insights`.
    strategy:
        Optional output of :func:`generate_campaign_strategy`.

    Returns
    -------
    str
        Formatted text report.
    """
    lines: list[str] = [
        "=" * 60,
        "  EMOTION DETECTION – MARKETING INSIGHTS REPORT",
        "=" * 60,
        "",
        f"  Dominant Emotion     : {insights.get('dominant_emotion', 'N/A').capitalize()}",
        f"  Overall Sentiment    : {insights.get('sentiment', 'N/A').capitalize()}",
        f"  Engagement Potential : {insights.get('engagement_potential', 'N/A').capitalize()}",
        f"  Audience Segment     : {insights.get('audience_segment', 'N/A')}",
        "",
        "  NARRATIVE",
        "  ---------",
        f"  {insights.get('narrative', '')}",
        "",
        "  EMOTION BREAKDOWN",
        "  -----------------",
    ]

    for emotion, score in sorted(
        insights.get("emotion_breakdown", {}).items(),
        key=lambda x: -x[1],
    ):
        bar = "█" * int(score / 5)
        lines.append(f"  {emotion:<10} {score:5.1f}%  {bar}")

    lines += [
        "",
        "  CONTENT RECOMMENDATIONS",
        "  -----------------------",
    ]
    for i, rec in enumerate(insights.get("content_recommendations", []), start=1):
        lines.append(f"  {i}. {rec}")

    if strategy:
        lines += [
            "",
            "  CAMPAIGN STRATEGY",
            "  -----------------",
            f"  Goal          : {strategy.get('campaign_goal', 'N/A')}",
            f"  Tone          : {strategy.get('recommended_tone', 'N/A')}",
            f"  Risk Level    : {strategy.get('risk_level', 'N/A').capitalize()}",
            "",
            "  Suggested Content Types:",
        ]
        for ct in strategy.get("content_types", []):
            lines.append(f"    • {ct}")
        lines += ["", "  Key Performance Indicators:"]
        for kpi in strategy.get("kpis", []):
            lines.append(f"    • {kpi}")
        lines += ["", "  Action Items:"]
        for i, action in enumerate(strategy.get("action_items", []), start=1):
            lines.append(f"    {i}. {action}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
