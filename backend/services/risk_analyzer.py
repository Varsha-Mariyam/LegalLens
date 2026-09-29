"""Risk analysis (guide section 12): transparent rule signals + optional LLM assessment.

Every rule produces evidence (what was found, why it matters, the reference it relies on). The rules run
for every clause. When an LLM is configured, it receives the clause, its explanation, the RAG references,
the reference clause and the rule evidence and returns a level, reason and precaution; the final level is
the higher of the rule level and the LLM level (conservative). Wording is always "potential risk".
"""
from __future__ import annotations

import re

from backend.services import legal_parsing as lp
from backend.services.llm_service import SYSTEM_PROMPT, LLMError, get_client, parse_json

LEVELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ICA = "Indian Contract Act, 1872"


def _ev(rule, level, finding, reason, precaution, source):
    return {"rule": rule, "level": level, "finding": finding, "reason": reason, "precaution": precaution,
            "source": source}


def _has(text, *patterns):
    return lp.contains_any(text, patterns)


def reference_notice_days(standard_text: str | None) -> float | None:
    if not standard_text:
        return None
    periods = lp.notice_periods(standard_text)
    return min((p["days"] for p in periods), default=None)


def rule_signals(clause: dict, doc_type: str, standard_text: str | None, doc_facts: dict) -> list[dict]:
    t = clause["text"]
    low = t.lower()
    cat = clause["category"]
    ev: list[dict] = []
    std_cite = "Project reference template (not an official legal standard)"

    # --- termination / notice ---
    notices = lp.notice_periods(t) if ("terminat" in low or "resign" in low or "vacat" in low or cat == "Termination") else []
    if notices:
        ref_days = reference_notice_days(standard_text) or 30.0
        shortest = min(notices, key=lambda n: n["days"])
        if shortest["days"] < ref_days:
            level = "HIGH" if shortest["days"] <= ref_days / 2 else "MEDIUM"
            ev.append(_ev("short_notice", level, f"Notice period of {shortest['text']} ({shortest['party'] or 'a party'})",
                          f"The notice period is shorter than the {int(ref_days)} days used in the reference clause, "
                          "leaving little time to respond or find alternatives.",
                          f"Ask for at least {int(ref_days)} days' written notice, applying equally to both parties.",
                          std_cite))
        distinct = sorted({n["days"] for n in notices})
        if len(distinct) > 1 and distinct[-1] >= 2 * distinct[0] and not re.search(r"either party|both parties|each party", low):
            ev.append(_ev("asymmetric_notice", "MEDIUM", "Different notice periods: " + " vs ".join(n["text"] for n in notices[:3]),
                          "The parties are bound by very different notice periods, so the clause is not mutual.",
                          "Request the same notice period for both parties.", std_cite))
    if re.search(r"terminat", low) and _has(t, r"without (any )?(prior )?notice", r"at any time without", r"with immediate effect"):
        cause = _has(t, r"misconduct", r"material breach", r"for cause", r"fraud")
        ev.append(_ev("termination_without_notice", "MEDIUM" if cause else "HIGH",
                      "Termination possible without notice" + (" (limited to cause)" if cause else ""),
                      "One party can end the relationship immediately, without time to cure or prepare.",
                      "Limit immediate termination to clearly defined serious cause, with a chance to respond.",
                      f"{ICA}, s.23; Brojo Nath Ganguly (1986)"))
    if cat == "Termination" and re.search(r"\b(company|employer|landlord|client|disclosing party|licensor|lessor)\s+may\s+terminate", low) \
            and not re.search(r"either party|both parties|each party|the (employee|tenant|receiving party|service provider) may terminate", low):
        ev.append(_ev("one_sided_termination", "MEDIUM", "Only one party is given a right to terminate",
                      "The termination right is one-sided.",
                      "Ask for a matching termination right for the other party.", f"{ICA}, s.23 (unfair terms)"))

    # --- liability ---
    capped = _has(t, r"shall not exceed", r"limited to", r"aggregate liability", r"maximum liability", r"\bcap(ped)?\b")
    if _has(t, r"\bunlimited\b", r"without (any )?limit", r"any and all (losses|damages|claims)", r"all losses") and not capped:
        ev.append(_ev("unlimited_liability", "HIGH", _has(t, r"\bunlimited\b", r"without (any )?limit", r"any and all (losses|damages|claims)", r"all losses"),
                      "Liability is open-ended with no upper limit.",
                      "Negotiate a liability cap and limit liability to losses caused by your own breach or negligence.",
                      f"{ICA}, s.73"))
    indirect = _has(t, r"consequential", r"indirect (loss|damage)", r"loss of profits?")
    if indirect and "liab" in low:
        if re.search(r"(not|no)\s+(be\s+)?(liable|responsible)[^.]{0,80}(consequential|indirect)|in no event", low):
            ev.append(_ev("indirect_loss_excluded", "LOW", "Indirect/consequential loss is excluded",
                          "Excluding indirect loss matches the default position in section 73.", "No action needed.",
                          f"{ICA}, s.73"))
        else:
            ev.append(_ev("indirect_loss_included", "HIGH", f"Liability includes '{indirect}'",
                          "Section 73 does not allow recovery of remote or indirect loss; this clause goes further.",
                          "Exclude indirect and consequential loss and loss of profit.", f"{ICA}, s.73"))
    if capped and cat == "Liability":
        ev.append(_ev("liability_cap", "LOW", f"Liability limited ('{capped}')", "A cap limits financial exposure.",
                      "Check that the cap amount is reasonable.", "General drafting practice"))
    if _has(t, r"indemnify") and _has(t, r"any and all", r"whatever (their|the) cause", r"all claims") and not _has(t, r"negligence", r"breach"):
        ev.append(_ev("broad_indemnity", "HIGH", "Broad indemnity not tied to own breach or negligence",
                      "The indemnifying party must cover losses even when it is not at fault.",
                      "Limit the indemnity to losses caused by your own breach, negligence or infringement; make it mutual.",
                      f"{ICA}, ss.124-125"))
    if _has(t, r"\bpenalty\b", r"liquidated damages", r"\bforfeit"):
        amounts = lp.extract_amounts(t)
        ev.append(_ev("penalty_or_forfeiture", "MEDIUM",
                      "Penalty / fixed damages / forfeiture" + (f" of {amounts[0]['text']}" if amounts else ""),
                      "Only reasonable compensation up to the named amount is recoverable; a penalty is not automatic.",
                      "Check that the amount reflects a genuine estimate of loss (e.g. actual training cost) and reduces over time.",
                      f"{ICA}, s.74; Kailash Nath Associates v. DDA (2015)"))

    # --- payment ---
    if _has(t, r"withh[oe]ld[^.]{0,60}(salary|payment|wages|fees|amount)", r"(salary|payment|wages|fees)[^.]{0,60}withh[oe]ld",
            r"may withhold"):
        level = "HIGH" if re.search(r"salary|wages", low) else "MEDIUM"
        ev.append(_ev("payment_withholding", level, "Payment can be withheld",
                      "Earned money can be withheld at one party's discretion.",
                      "Remove discretionary withholding; allow only deductions permitted by law or disputed amounts with reasons.",
                      "Payment of Wages Act, 1936, s.7 (authorised deductions)" if level == "HIGH" else "General drafting practice"))
    if cat == "Payment" and lp.extract_amounts(t) and not lp.extract_durations(t) and not re.search(
            r"\b(\d{1,2}(st|nd|rd|th)|monthly|every month|each month|per month|within|on or before|last working day)\b", low):
        ev.append(_ev("payment_no_timeline", "MEDIUM", "Amount stated without a payment date or period",
                      "Without a due date it is unclear when payment becomes late.",
                      "Add a clear due date or payment period.", "General drafting practice"))
    if doc_type == "service":
        periods = [d for d in lp.extract_durations(t) if d["unit"] == "day"]
        if cat == "Payment" and periods and max(p["days"] for p in periods) > 45 and re.search(r"invoice|payment|paid", low):
            ev.append(_ev("long_payment_period", "MEDIUM", f"Payment period of {max(p['days'] for p in periods):g} days",
                          "Payment period exceeds 45 days; for micro and small enterprises the MSMED Act caps it at 45 days.",
                          "Ask for payment within 30-45 days of invoice.", "MSMED Act, 2006, s.15"))
    if _has(t, r"increase the rent[^.]{0,60}(at any time|sole discretion)", r"(revise|increase)[^.]{0,40}at (his|her|its|their) (sole )?discretion"):
        ev.append(_ev("unilateral_price_change", "MEDIUM", "Rent or price can be increased unilaterally",
                      "One party can change the price at will.", "Fix the rent for the term; allow revision only on renewal by mutual agreement.",
                      "Model Tenancy Act, 2021 (rent revision as per agreement)"))

    # --- rental specifics ---
    if doc_type == "rental" and re.search(r"deposit", low):
        dep = lp.extract_amounts(t)
        rent = doc_facts.get("monthly_rent")
        if dep and rent:
            months = dep[0]["value"] / rent
            if months > 2.0:
                ev.append(_ev("excess_deposit", "MEDIUM", f"Deposit of {dep[0]['text']} ≈ {months:.1f} months' rent",
                              "The deposit exceeds the two-months'-rent benchmark in the Model Tenancy Act for residential premises.",
                              "Ask to reduce the deposit to two months' rent and fix a refund deadline.",
                              "Model Tenancy Act, 2021, s.11"))
        if _has(t, r"non-?refundable"):
            ev.append(_ev("non_refundable_deposit", "HIGH", "Deposit (or part of it) is non-refundable",
                          "A security deposit is meant to be refunded on handing over possession, after due deductions.",
                          "Make the deposit fully refundable, less itemised deductions.", "Model Tenancy Act, 2021, s.11"))
    if _has(t, r"lock-?in") and _has(t, r"forfeit"):
        ev.append(_ev("lockin_forfeiture", "HIGH", "Leaving during the lock-in forfeits the deposit",
                      "Losing the whole deposit is likely disproportionate to the actual loss.",
                      "Replace forfeiture with a notice period or a limited compensation amount.", f"{ICA}, s.74"))
    if doc_type == "rental" and _has(t, r"enter[^.]{0,40}(at any time|without (any )?notice)"):
        ev.append(_ev("entry_without_notice", "MEDIUM", "Landlord may enter without notice",
                      "The Model Tenancy Act benchmark requires at least 24 hours' prior notice for entry.",
                      "Require 24 hours' written notice except in emergencies.", "Model Tenancy Act, 2021"))
    if doc_type == "rental" and _has(t, r"all repairs", r"structural repairs") and _has(t, r"tenant shall", r"tenant must", r"tenant is"):
        ev.append(_ev("tenant_structural_repairs", "MEDIUM", "Tenant responsible for all or structural repairs",
                      "Structural repairs are normally the landlord's responsibility.",
                      "Limit the tenant's responsibility to minor repairs and damage caused by the tenant.",
                      "Model Tenancy Act, 2021"))

    # --- confidentiality / restraint ---
    if _has(t, r"in perpetuity", r"\bperpetual", r"indefinitely", r"at any time thereafter", r"forever"):
        ev.append(_ev("perpetual_obligation", "MEDIUM", "Obligation with no end date",
                      "Ordinary confidentiality obligations normally last for a defined period (trade secrets excepted).",
                      "Limit the obligation to a defined period (for example 2-3 years) except for trade secrets.",
                      "General drafting practice"))
    if cat == "Confidentiality" and _has(t, r"all information", r"any information whatsoever", r"information of any kind") \
            and not _has(t, r"public", r"already known", r"independently developed", r"required by law"):
        ev.append(_ev("broad_confidentiality", "MEDIUM", "Very broad definition with no exclusions",
                      "Without the usual exclusions even public information is covered.",
                      "Add exclusions for public, already-known, independently developed and legally required disclosures.",
                      "General drafting practice"))
    if _has(t, r"non-?compete", r"shall not (join|work for|engage|provide services to)[^.]{0,80}competit", r"competing business",
            r"not (join|start)[^.]{0,40}(competitor|competing)"):
        after = _has(t, r"after (leaving|termination|the termination|the end|expiry)", r"following (termination|the end)",
                     r"post[- ]termination", r"after the termination")
        if after:
            ev.append(_ev("post_term_restraint", "HIGH", "Restraint after the contract ends",
                          "A restraint on carrying on a trade, profession or business after the contract ends is void to that extent in India.",
                          "Remove the post-termination non-compete or limit it to non-solicitation and protecting confidential information.",
                          f"{ICA}, s.27; Superintendence Co. v. Krishan Murgai (1980)"))
    if doc_type == "employment" and _has(t, r"training (cost|bond)", r"service bond", r"leaves within"):
        ev.append(_ev("training_bond", "MEDIUM", "Training/service bond",
                      "Recovery is limited to reasonable compensation; a flat penalty unrelated to actual cost can be challenged.",
                      "Ask for the amount to reflect actual training cost and reduce proportionately with service.",
                      f"{ICA}, s.74"))

    # --- dispute ---
    if _has(t, r"arbitrator[^.]{0,40}(appointed|nominated|chosen|selected) (solely )?by the (company|employer|landlord|disclosing party|client|licensor|lessor)"):
        ev.append(_ev("unilateral_arbitrator", "HIGH", "Arbitrator chosen by one party only",
                      "A party interested in the outcome cannot unilaterally appoint the sole arbitrator.",
                      "Provide for appointment by mutual agreement, or by the court or an institution failing agreement.",
                      "Arbitration and Conciliation Act, 1996, s.12(5); Perkins Eastman v. HSCC (2019)"))
    if _has(t, r"waives? (any|all|its|his|her|their)? ?(right|rights) to (approach|sue|go to)", r"shall not (approach|file|sue)",
            r"waives any right to approach"):
        ev.append(_ev("waiver_of_legal_remedy", "HIGH", "Right to approach courts is waived",
                      "An agreement absolutely restricting a party from enforcing rights in ordinary tribunals is void (except arbitration).",
                      "Remove the waiver; keep access to courts or a fair arbitration.", f"{ICA}, s.28"))
    if cat == "Dispute" and _has(t, r"exclusive jurisdiction", r"courts at", r"courts of", r"seat of arbitration", r"venue"):
        ev.append(_ev("forum_selected", "LOW", "A specific court or seat is chosen",
                      "Choice of a competent court is generally valid; check that the place is convenient for you.",
                      "Confirm the chosen place is practical for you.", "Swastik Gases v. IOC (2013)"))

    # --- amendment / renewal ---
    if _has(t, r"(reserves the right|may) (to )?(amend|modify|change|vary|alter)[^.]{0,80}(at any time|without notice|sole discretion)"):
        ev.append(_ev("unilateral_amendment", "HIGH", "Terms can be changed unilaterally",
                      "One party can change the agreed terms without the other's consent.",
                      "Require any amendment to be in writing and signed by both parties.", f"{ICA}, ss.10 and 23"))
    if _has(t, r"automatically renew", r"auto-?renew", r"renew automatically"):
        ev.append(_ev("auto_renewal", "MEDIUM", "Automatic renewal",
                      "The contract continues unless someone acts in time.", "Note the renewal date and the notice needed to stop renewal.",
                      "General drafting practice"))
    return ev


def _combine(evidence: list[dict]) -> tuple[str, str, str]:
    negatives = [e for e in evidence if e["level"] != "LOW"]
    if not negatives:
        positives = [e for e in evidence if e["level"] == "LOW"]
        reason = " ".join(e["reason"] for e in positives) or "No potential risk indicators were found by the rule checks."
        return "LOW", reason, "Read the clause carefully and confirm it matches what was agreed."
    level = max(negatives, key=lambda e: LEVELS[e["level"]])["level"]
    ordered = sorted(negatives, key=lambda e: -LEVELS[e["level"]])
    reason = " ".join(f"{e['finding']}: {e['reason']}" for e in ordered[:3])
    precaution = " ".join(dict.fromkeys(e["precaution"] for e in ordered[:3]))
    return level, reason, precaution


def analyze(clause: dict, doc_type: str, standard: dict | None, references: list[dict], doc_facts: dict) -> dict:
    evidence = rule_signals(clause, doc_type, standard["text"] if standard else None, doc_facts)
    level, reason, precaution = _combine(evidence)
    result = {"risk_level": level, "risk_reason": reason, "precaution": precaution, "evidence": evidence,
              "method": "rule-based risk signals"}
    client = get_client()
    if not client.available:
        return result
    refs = "\n".join(f"- {r['metadata'].get('title')} [{r['metadata'].get('citation', '')}]: {r['text'][:500]}" for r in references[:2])
    ev_text = "\n".join(f"- {e['level']}: {e['finding']} — {e['reason']} ({e['source']})" for e in evidence) or "(none)"
    prompt = (f"Document type: {doc_type}\nOriginal clause:\n\"\"\"\n{clause['text']}\n\"\"\"\n"
              f"Clause summary: {clause.get('summary', '')}\n\nReference knowledge:\n{refs or '(none)'}\n\n"
              f"Project reference clause (not an official standard):\n{standard['text'] if standard else '(none)'}\n\n"
              f"Rule-based signals already detected:\n{ev_text}\n\n"
              "Assess the potential risk of this clause for the weaker party. Respond with JSON only: "
              '{"risk_level": "LOW|MEDIUM|HIGH", "reason": "evidence-based explanation in 1-3 sentences", '
              '"precaution": "1-2 practical precautions"}')
    try:
        data = parse_json(client.generate(SYSTEM_PROMPT, prompt, max_tokens=400))
        llm_level = str(data.get("risk_level", "")).upper()
        if llm_level not in LEVELS:
            raise LLMError(f"invalid risk level {llm_level!r}")
        final = llm_level if LEVELS[llm_level] >= LEVELS[level] else level
        result.update({"risk_level": final, "risk_reason": data.get("reason") or reason,
                       "precaution": data.get("precaution") or precaution,
                       "method": f"{client.label} + rule signals (final = higher of the two)"})
    except LLMError as exc:
        result["method"] = f"rule-based risk signals (LLM error: {str(exc)[:80]})"
    return result
