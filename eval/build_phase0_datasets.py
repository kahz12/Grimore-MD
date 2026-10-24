"""Build the tuning and held-out evaluation fixtures (eval/phase0_*.yaml).

The compact tuples below are the reviewable source of truth. Generated YAML is
checked in so running ``grimore eval`` never depends on this script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CORPUS_VERSION = "phase0-test-vault-2026-09-22"


def evidence(source: str, passage: str, locator: str) -> list[dict[str, str]]:
    return [{"source": source, "passage": passage, "locator": locator}]


def turn(
    case_id: str,
    question: str,
    answer: str,
    keywords: list[str],
    source: str,
    passage: str,
    locator: str,
    *,
    case_type: str | None = None,
    sources: list[str] | None = None,
    evidence_items: list[dict[str, str]] | None = None,
    review: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": case_id,
        "question": question,
        "expected_sources": sources or [source],
        "expected_keywords": keywords,
        "expected_answer": answer,
        "expected_evidence": evidence_items or evidence(source, passage, locator),
        "review_criteria": review or f"Debe comunicar correctamente: {answer}",
    }
    if case_type:
        item["case_type"] = case_type
    return item


def pair(
    *,
    split: str,
    language: str,
    source: str,
    fmt: str,
    difficulty: str,
    parent: dict[str, Any],
    follow_up: dict[str, Any],
    case_type: str = "direct",
    user_task: str = "find-fact",
    formats: list[str] | None = None,
) -> dict[str, Any]:
    parent = dict(parent)
    parent.update({
        "category": case_type,
        "language": language,
        "case_type": case_type,
        "document_formats": formats or [fmt],
        "user_task": user_task,
        "difficulty": difficulty,
        "split": split,
        "follow_ups": [follow_up],
    })
    return parent


AI = "Artificial Intelligence"
APPLE = "Apples Malus"
ARCH = "Arquitectura de Software"
ASTRO = "Astronomía General"
SOCRATES = "Sócrates"
FROGS = "Ranas"
GOTHIC = "gothic_architecture"
KOTLIN = "Kotlin Fundamentals"
ROME = "Roman Empire"
SOCIAL = "Social Engineering"
WETLAND = "Humedal La Esperanza"
WETLAND_REV = "Acta de revisión Humedal La Esperanza"


PAIRS: list[dict[str, Any]] = []


def add_doc_pairs(
    *, language: str, source: str, fmt: str, rows: list[tuple[dict, dict]],
    difficulty: str = "long-document",
) -> None:
    for index, (parent, follow_up) in enumerate(rows):
        PAIRS.append(pair(
            split="tuning" if index < 2 else "evaluation",
            language=language, source=source, fmt=fmt, difficulty=difficulty,
            parent=parent, follow_up=follow_up,
            case_type=parent.pop("_case_type", "direct"),
            user_task=parent.pop("_user_task", "find-fact"),
        ))


add_doc_pairs(language="en", source=AI, fmt="md", rows=[
    (
        turn("ai-dartmouth", "When and where was AI officially named as a field?",
             "At the Dartmouth Summer Research Project on Artificial Intelligence in 1956.",
             ["Dartmouth", "1956"], AI,
             "The field of AI was officially christened at the Dartmouth Summer Research Project on Artificial Intelligence in 1956",
             "The Birth and Golden Years"),
        turn("ai-dartmouth-organizers", "Who organized that project?",
             "John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon.",
             ["McCarthy", "Minsky", "Rochester", "Shannon"], AI,
             "organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon",
             "The Birth and Golden Years"),
    ),
    (
        {**turn("ai-first-winter", "What limitation did the Lighthill Report identify?",
             "It said combinatorial explosion kept AI techniques from solving real-world problems.",
             ["combinatorial explosion"], AI,
             "AI techniques had failed to solve real-world problems due to \"combinatorial explosion\"",
             "The First AI Winter"), "_case_type": "ambiguous"},
        turn("ai-perceptron-followup", "And which non-linear problem could a single-layer perceptron not solve?",
             "The exclusive-OR (XOR) problem.", ["XOR"], AI,
             "single-layer networks could not solve non-linearly separable functions, such as the exclusive-OR (XOR) problem",
             "The First AI Winter"),
    ),
    (
        turn("ai-transformer-origin", "Which 2017 paper introduced the Transformer?",
             "Vaswani et al.'s Attention Is All You Need.", ["Attention Is All You Need"], AI,
             "In June 2017, Vaswani et al. published the paper \"Attention Is All You Need\", introducing the Transformer",
             "The Generative Era"),
        turn("ai-gpt3-scale-followup", "How large was GPT-3 according to the same account?",
             "GPT-3 had 175 billion parameters.", ["175 billion"], AI,
             "GPT-3 (2020): Showed emergent behaviors, demonstrating that simply scaling model size (175 billion parameters)",
             "Large Language Models"),
    ),
])

add_doc_pairs(language="en", source=APPLE, fmt="md", rows=[
    (
        turn("apple-water-share", "What share of a mature apple's fresh weight is water?",
             "Approximately 84% to 86%.", ["84", "86"], APPLE,
             "Water accounts for approximately 84% to 86% of the fresh weight of a mature apple",
             "Water and Turgor Pressure"),
        turn("apple-turgor-followup", "Why does that water make the fruit crisp?",
             "Water in vacuoles creates turgor pressure against the cell walls, producing crispness.",
             ["turgor", "crisp"], APPLE,
             "This internal pressure, known as turgor pressure, is chemically responsible for the crispness and crunchiness of the fruit",
             "Water and Turgor Pressure"),
    ),
    (
        turn("apple-main-sugar", "Which soluble sugar accounts for the largest share in apples?",
             "Fructose, at roughly 50% to 60% of total soluble sugars.", ["Fructose", "50", "60"], APPLE,
             "Fructose | $C_6H_{12}O_6$ (Hexose) | 1.2 – 1.8 (Temperature dependent) | 50% – 60%",
             "Sugar Profile"),
        turn("apple-acid-followup", "And which organic acid dominates?",
             "L-malic acid makes up more than 90% of the total organic acid content.",
             ["malic", "90"], APPLE,
             "Over 90% of the total organic acid content in apples consists of L-malic acid",
             "Malic Acid"),
    ),
    (
        turn("apple-browning-components", "What three components are required for enzymatic browning?",
             "Oxygen, phenolic substrates, and polyphenol oxidase (PPO).",
             ["oxygen", "phenolic", "PPO"], APPLE,
             "The reaction requires three components: oxygen ($O_2$), phenolic substrates (specifically o-diphenols), and the enzyme Polyphenol Oxidase (PPO)",
             "Enzymatic Browning"),
        turn("apple-1mcp-followup", "How does 1-MCP delay ripening?",
             "It binds irreversibly to ethylene receptors and blocks ethylene signaling.",
             ["ethylene receptors", "blocking"], APPLE,
             "1-MCP binds irreversibly to ethylene receptors (ETR1 and others) on the cell membrane, blocking ethylene from binding",
             "1-Methylcyclopropene"),
    ),
])

add_doc_pairs(language="es", source=ARCH, fmt="md", rows=[
    (
        turn("arch-traditional-monolith", "¿Cómo se despliega un monolito tradicional?",
             "Toda la aplicación se empaqueta y despliega como una sola unidad.",
             ["única unidad"], ARCH,
             "toda la aplicación (interfaz de usuario, lógica de negocio y acceso a datos) se empaqueta y despliega como una única unidad indisoluble",
             "Monolito Tradicional"),
        turn("arch-monolith-disadvantages", "¿Y cuáles son dos desventajas al crecer?",
             "Es difícil de escalar y el acoplamiento puede hacer que un error derribe todo el sistema.",
             ["escalar", "acoplamiento"], ARCH,
             "Difícil de escalar vertical y horizontalmente, acoplamiento extremo (un error bota todo el sistema)",
             "Monolito Tradicional"),
    ),
    (
        turn("arch-microservices-properties", "¿Qué independencia ofrecen los microservicios?",
             "Cada servicio puede desplegarse de forma independiente y mantiene su propia base de datos.",
             ["independiente", "base de datos"], ARCH,
             "Despliegue 100% independiente, base de datos privada por servicio",
             "Microservicios"),
        turn("arch-microservices-cost", "¿Qué coste operativo introduce esa independencia?",
             "Complejidad operacional, latencia de red y transacciones distribuidas difíciles.",
             ["complejidad operacional", "latencia"], ARCH,
             "Complejidad operacional masiva (necesitas Kubernetes, CI/CD maduro, Service Mesh, observabilidad/trazabilidad), problemas por latencia de red",
             "Microservicios"),
    ),
    (
        {**turn("arch-kappa-meaning", "En la nota de arquitectura, ¿qué simplifica Kappa?",
                "Unifica el procesamiento: todo se trata como un flujo de eventos en tiempo real.",
                ["stream", "tiempo real"], ARCH,
                "Kappa: Simplifica la complejidad de Lambda unificándolo todo: todo es un stream de eventos en tiempo real",
                "Arquitectura Lambda y Kappa", case_type="similar-documents"),
         "_case_type": "similar-documents"},
        turn("arch-lambda-followup", "¿Qué dos capas conserva Lambda?",
             "Una Batch Layer histórica y una Speed Layer de tiempo real.",
             ["Batch Layer", "Speed Layer"], ARCH,
             "Lambda: Divide el procesamiento en dos capas. Una \"Batch Layer\" (lenta pero precisa, procesa TB de datos históricos) y una \"Speed Layer\"",
             "Arquitectura Lambda y Kappa"),
    ),
])

add_doc_pairs(language="es", source=ASTRO, fmt="md", rows=[
    (
        turn("astro-au-distance", "¿Qué distancia representa una unidad astronómica?",
             "La distancia media Tierra-Sol: aproximadamente 149.597.870,7 km.",
             ["Tierra", "Sol", "149"], ASTRO,
             "Unidad Astronómica (UA): Representa la distancia media entre la Tierra y el Sol",
             "Unidades de Medida"),
        turn("astro-parsec-followup", "¿Y cuántos años luz equivale un pársec?",
             "Aproximadamente 3,2616 años luz.", ["3.2616"], ASTRO,
             "1 \\text{ pc} \\approx 3.2616 \\text{ al}",
             "Unidades de Medida"),
    ),
    (
        turn("astro-kepler-first", "¿Qué afirma la primera ley de Kepler?",
             "Los planetas siguen órbitas elípticas con el Sol en uno de los focos.",
             ["elípticas", "focos"], ASTRO,
             "Todos los planetas se desplazan alrededor del Sol siguiendo órbitas elípticas, con el Sol situado en uno de los focos",
             "Las Tres Leyes de Kepler"),
        turn("astro-kepler-speed", "¿Qué implica la segunda ley para la velocidad orbital?",
             "Es máxima en el perihelio y mínima en el afelio.",
             ["perihelio", "afelio"], ASTRO,
             "la velocidad orbital de los planetas es variable (máxima en el perihelio y mínima en el afelio)",
             "Las Tres Leyes de Kepler"),
    ),
    (
        turn("astro-cmb-temperature", "¿Qué temperatura equivalente tiene hoy el CMB?",
             "Aproximadamente 2,725 K.", ["2.725"], ASTRO,
             "temperatura equivalente de 2.725 K",
             "Big Bang y CMB"),
        turn("astro-dark-energy-followup", "¿Qué fracción del universo atribuye el documento a la energía oscura?",
             "Aproximadamente 68,3%.", ["68.3"], ASTRO,
             "Energía Oscura ($\\Lambda$): Representa aproximadamente el 68.3% del universo",
             "Modelo Lambda-CDM"),
    ),
])

add_doc_pairs(language="es", source=SOCRATES, fmt="md", rows=[
    (
        turn("socrates-birth", "¿Cuándo y dónde nació Sócrates?",
             "Alrededor de 470 a.C. en el demo ateniense de Alopece.",
             ["470", "Alopece"], SOCRATES,
             "Sócrates nació alrededor del año 470 a.C. en el demo (distrito) de Alopece, al sur de Atenas",
             "Nacimiento y Linaje"),
        turn("socrates-mother-followup", "¿Qué profesión tenía su madre y qué método inspiró?",
             "Fainarate era partera y su oficio inspiró la mayéutica.",
             ["Fainarate", "partera", "mayéutica"], SOCRATES,
             "Fainarate, una partera u obstetra (maia). La profesión de su madre sirvió de inspiración directa para la concepción de su método filosófico, la mayéutica",
             "Nacimiento y Linaje"),
    ),
    (
        turn("socrates-campaigns", "¿En qué tres campañas militares combatió Sócrates?",
             "Potidea, Delio y Anfípolis.", ["Potidea", "Delio", "Anfípolis"], SOCRATES,
             "Combatió en tres campañas militares importantes",
             "El Ciudadano-Soldado"),
        turn("socrates-potidea-followup", "¿A quién rescató en Potidea?",
             "Al joven Alcibíades.", ["Alcibíades"], SOCRATES,
             "rescató al joven general Alcibíades cuando este cayó herido",
             "El Ciudadano-Soldado"),
    ),
    (
        {**turn("history-socrates-rome-dates", "Compara las fechas de la muerte de Sócrates y del fin tradicional del Imperio romano de Occidente.",
                "Sócrates murió en 399 a.C.; la deposición de Rómulo Augústulo ocurrió en 476 d.C.",
                ["399", "476"], SOCRATES,
                "399 a.C. : Juicio, condena e ingesta de cicuta",
                "Cronología", case_type="comparison", sources=[SOCRATES, ROME],
                evidence_items=evidence(SOCRATES, "399 a.C. : Juicio, condena e ingesta de cicuta", "Cronología") + evidence(ROME, "In 476 AD, the Germanic chieftain Odoacer deposed the last Western emperor, Romulus Augustulus", "Permanent Division and Fall")),
         "_case_type": "comparison", "_user_task": "compare-documents"},
        turn("history-agents-followup", "¿Qué agente o medio produjo cada desenlace?",
             "Sócrates fue ejecutado al beber cicuta; Odoacro depuso a Rómulo Augústulo.",
             ["cicuta", "Odoacer"], SOCRATES,
             "Sócrates bebió con serenidad el veneno preparado de la cicuta",
             "Los Últimos Días", case_type="comparison", sources=[SOCRATES, ROME],
             evidence_items=evidence(SOCRATES, "Sócrates bebió con serenidad el veneno preparado de la cicuta", "Los Últimos Días") + evidence(ROME, "Odoacer deposed the last Western emperor, Romulus Augustulus", "Permanent Division and Fall")),
    ),
])

add_doc_pairs(language="es", source=FROGS, fmt="md", rows=[
    (
        turn("frogs-anura-share", "¿Qué proporción de los anfibios vivos pertenece a Anura?",
             "Aproximadamente el 88%.", ["88"], FROGS,
             "El orden Anura es el más grande dentro de la clase Amphibia, abarcando aproximadamente el 88% de todas las especies de anfíbios vivos",
             "Clasificación Taxonómica"),
        turn("frogs-toads-followup", "¿La distinción entre rana y sapo tiene valor taxonómico estricto?",
             "No; es una distinción popular basada en rasgos y hábitos generales.",
             ["carece", "taxonómico"], FROGS,
             "Aunque popularmente se distingue entre \"rana\" y \"sapo\", esta distinción carece de un valor taxonómico estricto",
             "Ranas vs. Sapos"),
    ),
    (
        turn("frogs-respiration", "¿Qué tres mecanismos respiratorios usan las ranas adultas?",
             "Respiración cutánea, bucofaríngea y pulmonar.",
             ["cutánea", "bucofaríngea", "pulmonar"], FROGS,
             "Las ranas adultas emplean tres mecanismos complementarios para el intercambio gaseoso",
             "Sistema Respiratorio"),
        turn("frogs-positive-pressure", "¿Cómo ventilan los pulmones si no tienen diafragma ni costillas?",
             "Mediante bombeo de presión positiva con movimientos del piso de la boca.",
             ["presión positiva", "boca"], FROGS,
             "Al carecer de diafragma y costillas, las ranas respiran mediante bombeo de presión positiva",
             "Sistema Respiratorio"),
    ),
    (
        turn("frogs-metamorphosis-hormones", "¿Qué hormonas regulan la metamorfosis de los anuros?",
             "Las hormonas tiroideas tiroxina y triyodotironina.",
             ["tiroxina", "triyodotironina"], FROGS,
             "metamorfosis, regulada por las hormonas tiroideas (tiroxina y triyodotironina)",
             "Proceso de Metamorfosis"),
        turn("frogs-imago-followup", "¿Qué ocurre con la cola durante la fase de imago?",
             "Se reabsorbe por apoptosis y aporta energía durante el ayuno.",
             ["apoptosis", "energía"], FROGS,
             "Reabsorción de la cola por apoptosis",
             "Proceso de Metamorfosis"),
    ),
])

add_doc_pairs(language="en", source=GOTHIC, fmt="txt", rows=[
    (
        turn("gothic-origin-phase0", "Where and when did Gothic architecture originate?",
             "In 12th-century France, specifically Île-de-France around Paris.",
             ["France", "12th"], GOTHIC,
             "Originating in 12th-century France and lasting into the 16th century",
             "Introduction"),
        turn("gothic-saint-denis-followup", "Which building first incorporated its elements, and who rebuilt it?",
             "The Abbey of Saint-Denis, reconstructed by Abbot Suger.",
             ["Saint-Denis", "Suger"], GOTHIC,
             "The first major building to incorporate Gothic elements was the Abbey of Saint-Denis, reconstructed by Abbot Suger",
             "Historical Context"),
    ),
    (
        turn("gothic-trilogy-phase0", "What innovations form the Gothic Trilogy?",
             "The pointed arch, ribbed vault, and flying buttress.",
             ["pointed arch", "ribbed vault", "flying buttress"], GOTHIC,
             "The \"Gothic Trilogy\" of architectural innovations consists of",
             "Key Architectural Elements"),
        turn("gothic-buttress-followup", "Which one transfers lateral load away from the building?",
             "The flying buttress.", ["flying buttress"], GOTHIC,
             "transferring the lateral load away from the building",
             "Flying Buttress"),
    ),
    (
        turn("gothic-rayonnant", "What defined the Rayonnant Gothic phase?",
             "From about 1240-1350 it emphasized light and decorative tracery.",
             ["1240", "light", "tracery"], GOTHIC,
             "Rayonnant Gothic (c. 1240–1350): A shift toward even greater focus on light and decorative tracery",
             "Phases"),
        turn("gothic-flamboyant-followup", "What visual motif marked the following Flamboyant phase?",
             "Flame-like curves in tracery and highly ornate decoration.",
             ["flame-like", "tracery"], GOTHIC,
             "Marked by flame-like curves in tracery and highly ornate decoration",
             "Phases"),
    ),
])

add_doc_pairs(language="en", source=KOTLIN, fmt="md", rows=[
    (
        turn("kotlin-platforms", "Who designed Kotlin and which targets does it support?",
             "JetBrains designed it; it targets the JVM, JavaScript, and native machine code.",
             ["JetBrains", "JVM", "JavaScript", "Native"], KOTLIN,
             "Kotlin is a modern, cross-platform, statically typed programming language designed by JetBrains",
             "Introduction"),
        turn("kotlin-safety-followup", "Which common error does its safety pillar target?",
             "NullPointerException (NPE).", ["NullPointerException"], KOTLIN,
             "Safety: Focuses on eliminating common programming errors, most notably NullPointerException (NPE)",
             "Core Philosophy"),
    ),
    (
        turn("kotlin-val-var", "How do val and var differ in Kotlin?",
             "val is read-only after initialization; var is mutable.",
             ["read-only", "mutable"], KOTLIN,
             "val (Value): Defines a read-only variable. Once initialized, its reference cannot be reassigned",
             "Mutability"),
        turn("kotlin-elvis-followup", "Which operator supplies a fallback for null?",
             "The Elvis operator (?:).", ["Elvis"], KOTLIN,
             "Elvis Operator | str?.length ?: 0 | Provides a fallback value if the left-hand expression evaluates to null",
             "Null-Safety Operators"),
    ),
    (
        turn("kotlin-data-class", "What does Kotlin generate for a data class?",
             "equals, hashCode, toString, copy, and componentN functions.",
             ["equals", "hashCode", "copy", "componentN"], KOTLIN,
             "Kotlin automatically generates equals(), hashCode(), toString(), copy(), and componentN() functions",
             "Data Classes"),
        turn("kotlin-coroutine-followup", "How can a suspend function pause without blocking?",
             "It pauses the coroutine without blocking its underlying thread.",
             ["coroutine", "without blocking"], KOTLIN,
             "A function marked with suspend can pause the execution of a coroutine without blocking the thread it is running on",
             "Coroutine Core Concepts"),
    ),
])

add_doc_pairs(language="en", source=ROME, fmt="md", rows=[
    (
        turn("rome-transition", "Was Rome's transition from republic to empire sudden?",
             "No; it was a century-long process of decay, civil wars, and concentrated military power.",
             ["century-long", "civil wars"], ROME,
             "The transition from the Roman Republic to the Roman Empire was not a sudden event, but a century-long process of political decay, civil wars",
             "Fall of the Republic"),
        turn("rome-princeps-followup", "How did Octavian disguise his absolute power?",
             "He created the Principate and used the modest title Princeps, or First Citizen.",
             ["Principate", "Princeps"], ROME,
             "he established the Principate—a system where he retained absolute power under the modest title of Princeps (First Citizen)",
             "Augustan Settlement"),
    ),
    (
        turn("rome-territorial-peak", "Under which emperor did Rome reach its greatest territorial extent?",
             "Trajan, who reigned from 98 to 117 AD.", ["Trajan", "117"], ROME,
             "Under Trajan (reigned 98–117 AD), the empire reached its greatest geographic extent",
             "Nerva-Antonine Dynasty"),
        turn("rome-hadrian-followup", "How did his successor change policy?",
             "Hadrian reversed expansion and focused on securing borders.",
             ["Hadrian", "borders"], ROME,
             "Hadrian (reigned 117–138 AD) reversed the expansionist policy, focusing on securing the borders",
             "Nerva-Antonine Dynasty"),
    ),
    (
        turn("rome-concrete", "Why could Roman concrete set underwater?",
             "It mixed volcanic ash (pozzolana) with lime and water.",
             ["pozzolana", "lime"], ROME,
             "Made by mixing volcanic ash (pozzolana) with lime and water. It could set under water",
             "Key Technological Innovations"),
        turn("rome-aqueducts-followup", "How many aqueducts supplied peak Rome and how much water daily?",
             "Eleven aqueducts supplied over one million cubic metres daily.",
             ["11", "one million"], ROME,
             "the city of Rome had 11 aqueducts supplying over one million cubic meters of fresh water daily",
             "Key Technological Innovations"),
    ),
])

add_doc_pairs(language="en", source=SOCIAL, fmt="md", rows=[
    (
        turn("social-definition-phase0", "What does social engineering exploit in information security?",
             "It manipulates people by exploiting cognitive biases, trust, and psychological triggers.",
             ["cognitive biases", "trust"], SOCIAL,
             "social engineering exploits cognitive biases, trust, and psychological triggers",
             "Introduction"),
        turn("social-human-element-followup", "Why is it unlike a traditional cyberattack?",
             "It targets the human element rather than software vulnerabilities or network misconfiguration.",
             ["human element", "software vulnerabilities"], SOCIAL,
             "Unlike traditional cyberattacks that target software vulnerabilities or network misconfigurations, social engineering exploits cognitive biases",
             "Introduction"),
    ),
    (
        turn("social-authority", "How is the authority principle weaponized?",
             "Attackers impersonate executives, law enforcement, IT staff, or auditors.",
             ["Impersonating", "CEO"], SOCIAL,
             "Impersonating a CEO demanding urgent wire transfers or an external auditor demanding access",
             "Cialdini Principles"),
        turn("social-authority-defense", "What countermeasure does the guide recommend?",
             "Out-of-band verification that bypasses hierarchical pressure.",
             ["out-of-band", "hierarchical"], SOCIAL,
             "Establish out-of-band verification protocols that bypass hierarchical pressure",
             "Cialdini Principles"),
    ),
    (
        turn("social-spear-phishing", "How does spear phishing differ from mass phishing?",
             "It is highly targeted and customized with OSINT about a specific victim.",
             ["targeted", "OSINT"], SOCIAL,
             "Spear Phishing is a highly targeted version, customized using OSINT about the specific victim",
             "Phishing and Spear Phishing"),
        turn("social-fido-followup", "Why can FIDO2 resist a look-alike phishing domain?",
             "The authenticator binds its private key to the legitimate RP ID and refuses the spoofed origin.",
             ["RP ID", "refuses"], SOCIAL,
             "The authenticator permanently binds the private key to the RP ID (the domain)",
             "FIDO2 Origin Binding"),
    ),
])


# PDF pairs: first two tune extraction; the remaining three are held out.
pdf_rows = [
    (
        turn("wetland-start-area", "¿Cuándo comenzó el proyecto y qué área interviene?",
             "Comenzó el 14 de marzo de 2024 y cubre 37,5 hectáreas.",
             ["14 de marzo", "37,5"], WETLAND,
             "El proyecto de restauración del Humedal La Esperanza comenzó el 14 de marzo de 2024. El área de intervención aprobada es de 37,5 hectáreas",
             "p.1"),
        turn("wetland-target-followup", "¿Cuál era su meta operativa principal?",
             "Retirar el jacinto invasor de 8 hectáreas antes del 31 de diciembre de 2025.",
             ["8 hectáreas", "31 de diciembre"], WETLAND,
             "La meta operativa principal es retirar el jacinto de agua invasor de 8 hectáreas antes del 31 de diciembre de 2025",
             "p.1"),
    ),
    (
        turn("wetland-stations", "¿Cuáles son las estaciones de seguimiento y dónde están?",
             "Norte-01 está aguas arriba y Sur-02 aguas abajo.",
             ["Norte-01", "Sur-02"], WETLAND,
             "Norte-01, situada aguas arriba de la zona intervenida, y Sur-02, situada aguas abajo",
             "p.2"),
        turn("wetland-schedule-followup", "¿Cuándo se mide la calidad del agua?",
             "El primer martes de cada mes, con muestra extra tras lluvias superiores a 30 mm en 24 horas.",
             ["primer martes", "30 mm"], WETLAND,
             "La calidad del agua se mide el primer martes de cada mes",
             "p.2"),
    ),
    (
        turn("wetland-june-status", "¿Cuál era el estado del proyecto en junio de 2025?",
             "5,5 hectáreas despejadas, fósforo mediano de 0,12 mg/L, 16 aves y 5 anfibios.",
             ["5,5", "0,12", "16", "5"], WETLAND,
             "A junio de 2025 se habían despejado 5,5 de las 8 hectáreas previstas",
             "p.3"),
        turn("wetland-missing-data-followup", "¿Qué dato faltó y cómo debe tratarse?",
             "Sur-02 no tuvo dato de turbidez en mayo; no debe imputarse ni interpretarse como mejora.",
             ["Sur-02", "mayo", "no imputa"], WETLAND,
             "La estación Sur-02 quedó sin datos en mayo de 2025 por una avería del sensor de turbidez. El informe no imputa valores ni declara una mejora",
             "p.3"),
    ),
    (
        {**turn("wetland-review-threshold", "¿Cuándo se activa una revisión técnica por fósforo?",
             "Cuando supera 0,20 mg/L en dos mediciones mensuales consecutivas.",
             ["0,20", "dos"], WETLAND,
             "Se activa una revisión técnica si el fósforo total supera 0,20 mg/L en dos mediciones mensuales consecutivas",
             "p.3"), "_user_task": "review-note"},
        turn("wetland-replant-followup", "¿Qué ocurre si la supervivencia vegetal baja del 70% dos veces?",
             "Verde Vivo debe replantar el sector dentro de 30 días.",
             ["replantar", "30 días"], WETLAND,
             "la Cooperativa Verde Vivo debe replantar el sector afectado dentro de los siguientes 30 días",
             "p.3"),
    ),
    (
        {**turn("wetland-deadline-conflict", "¿Qué fecha límite dan las dos versiones para retirar el jacinto?",
                "El informe v1.2 dice 31 de diciembre de 2025; el acta posterior la corrige al 30 de noviembre de 2025.",
                ["31 de diciembre", "30 de noviembre"], WETLAND,
                "antes del 31 de diciembre de 2025", "p.1", case_type="contradictory",
                sources=[WETLAND, WETLAND_REV],
                evidence_items=evidence(WETLAND, "antes del 31 de diciembre de 2025", "p.1") + evidence(WETLAND_REV, "se adelanta del 31 de diciembre de 2025 al 30 de noviembre de 2025", "Cambios aprobados"),
                review="Debe presentar ambos valores, sus fechas o versiones y explicar que el acta posterior es la corrección aprobada."),
         "_case_type": "contradictory", "_user_task": "compare-documents"},
        turn("wetland-phosphorus-conflict", "¿Y qué meta de fósforo queda vigente?",
             "El informe decía menos de 0,10 mg/L; el acta posterior la endurece a menos de 0,08 mg/L.",
             ["0,10", "0,08"], WETLAND,
             "Menos de 0,10 mg/L", "p.1", case_type="contradictory",
             sources=[WETLAND, WETLAND_REV],
             evidence_items=evidence(WETLAND, "Menos de 0,10 mg/L", "p.1") + evidence(WETLAND_REV, "pasa de menos de 0,10 mg/L a menos de 0,08 mg/L en tres mediciones consecutivas", "Cambios aprobados"),
             review="Debe conservar la discrepancia y adoptar 0,08 mg/L como corrección posterior, sin ocultar el valor previo."),
    ),
]
for index, (parent, follow_up) in enumerate(pdf_rows):
    case_type = parent.pop("_case_type", "direct")
    user_task = parent.pop("_user_task", "find-fact")
    PAIRS.append(pair(
        split="tuning" if index < 2 else "evaluation", language="es",
        source=WETLAND, fmt="pdf", difficulty="extraction-hard",
        parent=parent, follow_up=follow_up, case_type=case_type,
        user_task=user_task,
        formats=["pdf", "md"] if case_type == "contradictory" else ["pdf"],
    ))


NEGATIVE_TUNING = [
    ("neg-es-clima-bogota", "¿Qué tiempo hará mañana en Bogotá?", "es"),
    ("neg-en-bitcoin-live", "What is Bitcoin's live price right now?", "en"),
    ("neg-es-receta-arepas", "¿Cuál es una receta de arepas para cuatro personas?", "es"),
    ("neg-en-train-london", "What time is the last train from London to Edinburgh tonight?", "en"),
    ("neg-es-presidente-actual", "¿Quién es el presidente actual de Chile?", "es"),
    ("neg-en-mars-weather", "What is today's weather at the Jezero crater on Mars?", "en"),
    ("neg-es-sensor-serial", "¿Cuál es el número de serie del sensor averiado de Sur-02?", "es"),
    ("neg-en-wetland-budget", "What was the exact approved budget for the wetland project?", "en"),
    ("neg-es-socrates-estatura", "¿Cuánto medía Sócrates exactamente?", "es"),
    ("neg-en-kotlin-release", "What is the latest stable Kotlin compiler version today?", "en"),
    ("neg-es-ranas-poblacion", "¿Cuántas ranas viven hoy en el Humedal La Esperanza?", "es"),
    ("neg-en-rome-weather", "What was the weather in Rome on 4 September 476 AD?", "en"),
    ("neg-es-manzana-precio", "¿Cuánto cuesta hoy un kilo de manzanas en Medellín?", "es"),
    ("neg-en-gothic-budget", "What was the exact construction budget of Amiens Cathedral?", "en"),
    ("neg-es-ai-acciones", "¿Qué acción tecnológica subirá más esta semana?", "es"),
    ("neg-en-private-email", "What is the private email address of the wetland project manager?", "en"),
]

NEGATIVE_EVALUATION = [
    ("neg-es-loteria", "¿Qué número ganará la próxima lotería?", "es"),
    ("neg-en-sourdough", "Give me the vault's sourdough starter recipe.", "en"),
    ("neg-es-vuelo", "¿Cuál es el vuelo más barato de Bogotá a Madrid mañana?", "es"),
    ("neg-en-football", "Who won yesterday's Premier League match?", "en"),
    ("neg-es-humedal-contrato", "¿Cuál es el número del contrato de Verde Vivo?", "es"),
    ("neg-en-ai-secret", "What proprietary dataset was used to train the local model?", "en"),
    ("neg-es-socrates-adn", "¿Cuál era la secuencia de ADN de Sócrates?", "es"),
    ("neg-en-apple-genome", "Which exact orchard supplied the apples analyzed in the note?", "en"),
    ("neg-es-ranas-gps", "¿Cuáles son las coordenadas GPS de cada rana censada?", "es"),
    ("neg-en-rome-population", "What was Rome's exact population on 1 January 100 AD?", "en"),
    ("neg-es-catedral-arquitecto", "¿Cuál fue el domicilio del maestro de obras de Chartres?", "es"),
    ("neg-en-security-incident", "Which employee clicked the latest phishing simulation?", "en"),
    ("neg-es-kotlin-salario", "¿Cuál es el salario medio actual de un programador Kotlin en Cali?", "es"),
    ("neg-en-telescope-password", "What is the administrator password for the James Webb telescope?", "en"),
]


def negative(case_id: str, question: str, language: str, split: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "question": question,
        "category": "negative",
        "negative": True,
        "language": language,
        "case_type": "unanswerable",
        "document_formats": ["md", "pdf", "txt"],
        "expected_answer": "abstain",
        "expected_keywords": [],
        "expected_evidence": [],
        "review_criteria": "Debe indicar que el corpus no contiene evidencia suficiente y no inventar el dato.",
        "user_task": "find-fact",
        "difficulty": "negative-near-corpus" if any(
            word in question.lower() for word in ("humedal", "wetland", "sócrates", "kotlin", "rome", "apple", "rana", "gothic", "ai")
        ) else "negative-out-of-domain",
        "split": split,
    }


def write_dataset(split: str, path: Path, negatives: list[tuple[str, str, str]]) -> None:
    questions = [item for item in PAIRS if item["split"] == split]
    questions.extend(negative(case_id, question, language, split)
                     for case_id, question, language in negatives)
    payload = {
        "version": 2,
        "set": split,
        "corpus_version": CORPUS_VERSION,
        "questions": questions,
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_dataset("tuning", ROOT / "phase0_tuning.yaml", NEGATIVE_TUNING)
    write_dataset("evaluation", ROOT / "phase0_evaluation.yaml", NEGATIVE_EVALUATION)
    tuning_turns = sum(1 + len(q.get("follow_ups", [])) for q in PAIRS if q["split"] == "tuning") + len(NEGATIVE_TUNING)
    eval_turns = sum(1 + len(q.get("follow_ups", [])) for q in PAIRS if q["split"] == "evaluation") + len(NEGATIVE_EVALUATION)
    assert tuning_turns == 60, tuning_turns
    assert eval_turns == 40, eval_turns
    print(f"wrote {tuning_turns} tuning turns and {eval_turns} evaluation turns")
