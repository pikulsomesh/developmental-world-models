/* Baby Senses Lab: data tables.
   Ages 0-12 months follow the capability matrix in the project proposal (sections 3.1-3.5) and the
   paper's filter tables (dwm/capability.py). Ages 15-24 months follow the CDC/AAP developmental
   milestone checklists (Zubler et al., 2022). Every value is a typical age for a population, with wide
   normal variation; none of it describes a particular child. */
window.LAB = (() => {
"use strict";

// ---------------------------------------------------------------- vision numbers
// "brain": proposal / paper table (largely visual evoked potential studies; Norcia & Tyler 1985).
// "look": behavioural acuity-card norms (Teller 1997; Mayer et al. 1995), centre of a range that
// spans about one octave (half to double) either side.
const ACUITY = {
  brain: { 0: 1.0, 1: 1.5, 2: 2.0, 3: 4.0, 4: 6.0, 5: 10.0, 6: 15.0, 9: 25.0, 12: 30.0 },
  look: { 0: 0.8, 1: 1.0, 2: 1.8, 3: 2.6, 4: 3.4, 6: 4.8, 9: 6.0, 12: 7.5, 18: 9.0, 24: 10.5, 60: 30 },
};
const CONTRAST_SENS = { 0: 2, 1: 3, 2: 5, 3: 10, 4: 20, 5: 30, 6: 50, 9: 80, 12: 100 };
const COLOR_SAT = { 0: 0.1, 1: 0.2, 2: 0.4, 3: 0.7, 4: 0.9, 5: 1.0 };

// ---------------------------------------------------------------- hearing (coarse; see guide)
// Extra quietness (dB) reflects infants' raised hearing thresholds (Werner 2007: roughly 15-30 dB
// above adults at 3 months, 10-15 dB by 6 months, closer still by 12). Stereo width follows the
// paper's binaural table; the noise floor stands for less selective listening.
const HEAR = {
  quieter_db: { 0: 25, 3: 20, 6: 12, 12: 6, 24: 3 },
  stereo: { 0: 0.3, 3: 0.5, 6: 0.8, 12: 1.0 },
  noise: { 0: 0.08, 3: 0.06, 6: 0.04, 12: 0.02, 24: 0.01 },
  treble_cut_db: { 0: 12, 1: 6, 3: 4, 6: 1, 12: 0 },
};

// ---------------------------------------------------------------- age profile, by domain
// Each row: [from-month, plain-language sentence]. The latest row at or below the chosen age applies.
const DOMAINS = [
  { id: "vision", name: "Seeing", color: "var(--c-vision)", items: [
    ["Sharpness", [[0, "Very blurry: roughly 20/400 to 20/800. Large, bold shapes at 20-30 cm are what stands out."], [2, "Still blurry (about 20/150), but improving quickly."], [3, "About 20/100. Medium-sized patterns become visible."], [4, "About 20/60: twice as sharp as at two months."], [6, "Near-adult on brain-response measures (about 20/30); looking tests still give lower numbers."], [12, "Adult-like on brain-response measures. Fine detail keeps sharpening into the preschool years."]]],
    ["Contrast", [[0, "Sees only strong light-dark differences, such as black on white."], [3, "Medium contrasts start to show up."], [6, "Most everyday contrasts are visible."], [12, "Close to adult contrast sensitivity."]]],
    ["Colour", [[0, "Mostly light and dark; strong reds may stand out."], [2, "Reds and greens begin to separate."], [3, "Most colours are distinguishable."], [5, "Colour vision is close to adult."]]],
    ["Movement", [[0, "Notices movement, especially large moving things."], [1, "Follows slowly moving objects with jerky eye movements."], [2, "Smooth tracking of a slowly moving toy begins."], [4, "Tracks in all directions."], [6, "Adult-like motion seeing."]]],
    ["Faces", [[0, "Prefers face-like patterns over scrambled ones."], [1, "Scans the outer edge of faces and the eyes."], [2, "Looks at inner features; social smiling begins."], [5, "Tells apart facial expressions."], [9, "Follows where someone is looking."], [12, "Shares attention: looks where you point and back at you."]]],
    ["Depth", [[0, "Little depth perception."], [3, "Two-eye (binocular) depth begins to emerge."], [4, "3D depth from both eyes develops quickly."], [6, "Uses depth well, for example when reaching."]]],
  ] },
  { id: "hearing", name: "Hearing and language", color: "var(--c-audio)", items: [
    ["Listening", [[0, "Hears from birth and prefers mother's voice, but quiet sounds are missed."], [3, "Prefers the rhythm of their home language."], [5, "Picks out the melody and stress patterns of speech."], [7, "Finds word boundaries in running speech."], [12, "Listens to speech in adult-like ways, though hearing in noise keeps improving for years."]]],
    ["Speech sounds", [[0, "Can tell apart speech sounds from every language."], [6, "Starts tuning in to the sounds of the home language."], [8, "Tuning continues; non-native sound contrasts get harder."], [10, "Mostly attends to home-language sound contrasts."]]],
    ["Understanding words", [[0, "No words yet, but already learning from voices."], [5, "Recognises own name."], [6, "Some babies show signs of knowing a few very common words (food, body parts)."], [8, "Understands a handful of words (about 5-20)."], [10, "Understands several dozen words (about 20-50)."], [12, "Understands about 50-100 words and simple requests with gestures."], [18, "Follows one-step directions without gestures."], [24, "Understands hundreds of words; points to things in a book when asked."]]],
    ["Talking", [[0, "Cries and makes small sounds."], [2, "Coos."], [6, "Takes turns making sounds; babbling begins."], [9, "Repeated babble like 'mamama', 'bababa'."], [12, "Says one or two words such as 'mama' or 'dada'."], [15, "Tries to say one or two words besides 'mama' and 'dada'."], [18, "Tries to say three or more words."], [24, "Puts two words together, like 'more milk'."]]],
    ["Locating sounds", [[0, "Turns roughly toward a sound, slowly."], [4, "Turns quickly toward sounds to the side."], [6, "Locates sounds more precisely."], [12, "Much more precise; still improving."]]],
  ] },
  { id: "touch", name: "Touch and body sense", color: "var(--c-touch)", items: [
    ["Touch", [[0, "The most mature sense at birth: responds to touch everywhere, strong rooting and grasp reflexes."], [3, "Explores with hands and mouth."], [6, "Explores textures deliberately with fingers."], [12, "Fine touch keeps sharpening."]]],
    ["Body position", [[0, "Senses body position from birth, but control is limited."], [3, "Uses body sense to start controlling head and arms."], [6, "Combines body sense and vision for reaching."], [12, "Uses body sense for balance when standing."]]],
  ] },
  { id: "thinking", name: "Thinking", color: "var(--c-curriculum)", items: [
    ["Hidden objects", [[0, "Out of sight seems out of mind."], [4, "Looks longer when a hidden object seems to vanish (lab studies; results vary)."], [5, "Tracks objects as they pass behind a screen."], [8, "Searches for a toy hidden under a cloth, but may look in the old place ('A-not-B' error)."], [11, "The A-not-B error fades."], [18, "Finds objects after several hidden moves."]]],
    ["Cause and effect", [[0, "No sense of cause yet."], [4, "Starts to expect that things move when pushed."], [5, "Sees one object launching another as causal."], [7, "Infers hidden causes; repeats actions to make things happen."], [12, "Experiments to see what happens."]]],
    ["Memory", [[0, "Remembers for seconds."], [3, "About 10 seconds of working memory."], [6, "About 20 seconds."], [8, "About 30 seconds."], [12, "About a minute; remembers routines for days."], [18, "Imitates actions seen earlier (deferred imitation)."]]],
    ["Categories and number", [[0, "Groups things that look alike; tells 1 from 2."], [4, "Forms simple categories by appearance; notices small set changes."], [6, "Understands above and below; adds and subtracts tiny sets in lab tasks."], [9, "Groups things by what they do."], [12, "Broad categories like 'animals' and 'vehicles'."]]],
  ] },
  { id: "social", name: "Social", color: "var(--c-grown)", items: [
    ["Eyes and gaze", [[0, "Prefers faces looking directly at them."], [3, "Follows a head turn."], [7, "Reliably follows gaze."], [9, "Follows gaze to the target object."]]],
    ["Goals of others", [[0, "None yet."], [3, "Notices goal-directed motion."], [5, "Expects people to act efficiently toward goals."], [7, "Consistently reads simple goals."], [11, "Reads multi-step goals."]]],
    ["Helping and hindering", [[0, "None yet."], [5, "Begins to prefer kind characters in puppet shows."], [7, "Prefers helpers over hinderers (lab studies)."], [12, "Evaluates hinderers."]]],
    ["Shared attention", [[0, "None yet."], [3, "Face-to-face back-and-forth."], [6, "Shared attention starts to emerge."], [9, "Looks between you and an object."], [12, "Shares interest by pointing and showing."], [18, "Points to show you something interesting."]]],
  ] },
  { id: "movement", name: "Movement", color: "var(--c-proprio)", items: [
    ["Head and body", [[0, "Lifts head briefly when on tummy."], [2, "Holds head up to about 45 degrees on tummy."], [4, "Full head control."], [6, "Sits with support, then in a tripod position."], [8, "Sits without support."]]],
    ["Hands", [[0, "Reflex grasp."], [2, "Swipes at objects; opens hands."], [4, "Reaches on purpose; whole-hand grasp."], [6, "Passes toys from hand to hand."], [8, "Pincer grasp (thumb and finger) emerging."], [10, "Picks up small things with thumb and finger."], [15, "Stacks two blocks."], [24, "Stacks several blocks; uses a spoon."]]],
    ["Getting around", [[0, "Stepping reflex when held upright."], [4, "Pre-crawling: rolls, pushes up."], [6, "Belly crawling may begin."], [8, "Crawls on hands and knees; pulls to stand."], [10, "Cruises along furniture; may take first steps."], [15, "Takes a few steps alone."], [18, "Walks without holding on."], [24, "Runs; kicks a ball."]]],
  ] },
];

// ---------------------------------------------------------------- abilities for activity checks
// Typical month at which an ability is about half-established, and how gradually it arrives
// (logistic width in months). Derived from the matrix rows above; deliberately coarse.
const CAPS = {
  visual_interest: [0.3, 0.5, "looking at bold patterns"], visual_attention: [1, 0.7, "paying visual attention"],
  smooth_pursuit: [2, 0.8, "following moving things with the eyes"], head_control: [3, 1, "holding the head steady"],
  face_recognition: [2, 1, "recognising faces"], social_engagement: [2.5, 1, "social back-and-forth"],
  auditory_attention: [1, 1, "turning toward sounds"], colour_vision: [3, 0.8, "seeing colours"],
  reaching: [4, 1, "reaching for things"], grasping: [4.5, 1, "grasping with the whole hand"],
  anticipation: [4, 1.2, "anticipating what comes next"], depth: [4.5, 1, "seeing depth"],
  rolling: [5, 1, "rolling over"], name_recognition: [5.5, 1, "recognising own name"],
  cause_effect: [6, 1.2, "understanding cause and effect"], sitting: [6.5, 1, "sitting up"],
  transfer: [6, 1, "passing toys hand to hand"], intentional_action: [7, 1.2, "acting on purpose"],
  object_permanence: [8, 1.5, "knowing hidden things still exist"], hand_eye: [7, 1.5, "hand-eye coordination"],
  crawling: [8.5, 1.2, "crawling"], action_imitation: [9, 1.5, "copying actions"], babbling: [7, 1.2, "babbling"],
  fine_motor: [10, 1.5, "thumb-and-finger (pincer) grasp"], word_comprehension: [10, 1.8, "understanding words"],
  joint_attention: [10, 1.5, "sharing attention with you"], pointing_comprehension: [11, 1.5, "following a pointing finger"],
  grasp_release: [11, 1.5, "letting go on purpose"], standing: [10, 1.2, "standing with support"],
  walking: [13, 1.5, "walking"], trial_error: [13, 2, "trial-and-error experimenting"],
  spatial_reasoning: [14, 2, "reasoning about space"], persistence: [14, 2, "sticking with a task"],
  shape_matching: [17, 2.5, "matching shapes"], problem_solving: [15, 2.5, "solving simple problems"],
  pretend_play: [17, 2, "pretend play"], self_recognition: [18, 2, "recognising self in a mirror"],
  two_word_speech: [22, 2.5, "two-word sentences"],
};
const level = (cap, age) => { const c = CAPS[cap]; return 1 / (1 + Math.exp(-(age - c[0]) / c[1])); };

// ---------------------------------------------------------------- activity library
// requires: ability -> level needed (0-1). typical / full: months when most babies engage / fully enjoy it.
const A = (id, name, cat, description, requires, typical, full, benefits, tips) => ({ id, name, cat, description, requires, typical, full, benefits, tips });
const ACTIVITIES = [
  A("contrast_cards", "High-contrast cards", "Seeing", "Bold black-and-white patterns held 20-30 cm from the face.", { visual_interest: 0.3 }, 0, 3, ["visual attention", "early focusing"], "Hold the card still at 20-30 cm; move it slowly once the baby is looking."),
  A("face_time", "Face-to-face time", "Social", "Holding the baby close and talking, smiling and making faces.", { visual_interest: 0.3, face_recognition: 0.2 }, 0, 4, ["bonding", "face learning", "turn-taking"], "At 20-30 cm your face is in the clearest range for a newborn."),
  A("singing", "Songs and lullabies", "Hearing", "Singing to the baby with exaggerated melody.", { auditory_attention: 0.3 }, 0, 12, ["listening", "rhythm of language", "calming"], "Babies prefer sung, sing-song speech with a slow tempo."),
  A("tummy_time", "Tummy time", "Movement", "Short, supervised periods lying on the tummy while awake.", { head_control: 0.1 }, 0, 4, ["neck strength", "head control"], "Start with a minute or two, several times a day, always supervised."),
  A("tracking_toy", "Track a moving toy", "Seeing", "Follow a slowly moving toy with the eyes.", { smooth_pursuit: 0.4, visual_attention: 0.3, head_control: 0.3 }, 2, 4, ["visual tracking", "attention", "eye-head coordination"], "Move slowly (about 30 cm per second); use high-contrast toys."),
  A("mirror_play", "Mirror play", "Social", "Looking at themselves in a safe mirror, touching it.", { face_recognition: 0.3, visual_interest: 0.3, reaching: 0.2 }, 2, 18, ["visual interest", "social", "self-awareness later"], "Young babies enjoy it but don't recognise themselves until about 15-18 months."),
  A("rattle", "Rattle or sound toy", "Hearing", "Shake a rattle to one side and watch the baby turn.", { auditory_attention: 0.4, head_control: 0.4 }, 2, 6, ["sound location", "attention"], "Shake gently and wait: young babies turn slowly."),
  A("hanging_toy", "Reach for a hanging toy", "Movement", "A toy hanging within arm's reach while lying on the back.", { reaching: 0.4, hand_eye: 0.2, depth: 0.3 }, 3, 6, ["reaching", "hand-eye coordination"], "Swiping comes before grabbing; both count."),
  A("texture_play", "Texture play", "Touch", "Touching fabrics, soft brushes and textured books.", { grasping: 0.3 }, 3, 9, ["touch discrimination", "exploration"], "Avoid small parts; everything goes in the mouth."),
  A("peek_a_boo", "Peek-a-boo", "Thinking", "A face disappears behind hands and reappears.", { object_permanence: 0.3, smooth_pursuit: 0.5, social_engagement: 0.4, anticipation: 0.2 }, 4, 8, ["object permanence", "social bonding", "anticipation"], "Start with short disappearances for younger babies."),
  A("respond_name", "Respond to name", "Hearing", "Turning the head when their name is called.", { name_recognition: 0.6, auditory_attention: 0.5, social_engagement: 0.5, head_control: 0.9 }, 6, 9, ["name recognition", "social responsiveness"], "Use an animated voice; call from the side, not straight in front."),
  A("cause_effect_toy", "Push-button toy", "Thinking", "Press a button or lever to cause a light, sound or movement.", { cause_effect: 0.4, intentional_action: 0.5, fine_motor: 0.1 }, 6, 9, ["cause and effect", "sense of agency"], "Clear, immediate feedback matters."),
  A("banging", "Banging and drumming", "Hearing", "Banging spoons on pots or a toy drum.", { grasping: 0.6, sitting: 0.4, cause_effect: 0.3 }, 6, 10, ["cause and effect", "rhythm", "arm control"], "Join in and copy the rhythm back."),
  A("water_play", "Splashing in water", "Touch", "Supervised splashing in a shallow tub.", { sitting: 0.6, cause_effect: 0.3 }, 6, 12, ["touch", "cause and effect"], "Never leave a baby alone near water, even for a moment."),
  A("hide_toy", "Find the hidden toy", "Thinking", "Hide a toy under a cloth while the baby watches.", { object_permanence: 0.5, reaching: 0.8, intentional_action: 0.5 }, 8, 12, ["object permanence", "memory"], "Expect the classic 'A-not-B' error around 8-10 months: it is normal."),
  A("pat_a_cake", "Pat-a-cake", "Social", "Clapping game with a song.", { action_imitation: 0.4, social_engagement: 0.6, sitting: 0.6 }, 7, 12, ["imitation", "rhythm", "turn-taking"], "Guide their hands at first."),
  A("point_named", "Point to a named object", "Hearing", "Point to an object when its name is spoken.", { word_comprehension: 0.6, joint_attention: 0.6, pointing_comprehension: 0.5, visual_attention: 0.9 }, 9, 12, ["vocabulary", "listening", "comprehension"], "Use high-interest objects (ball, dog, mama)."),
  A("imitation", "Copy the action (clap, wave)", "Social", "Copy simple actions like clapping or waving bye-bye.", { action_imitation: 0.5, social_engagement: 0.5, sitting: 0.6 }, 9, 12, ["imitation", "social learning", "motor planning"], "Start simple and exaggerate movements."),
  A("picture_book", "Picture books and naming", "Hearing", "Looking at board books and naming pictures.", { word_comprehension: 0.4, joint_attention: 0.4, colour_vision: 0.9 }, 6, 18, ["vocabulary", "shared attention"], "Name and point; let the baby turn the (board) pages."),
  A("roll_ball", "Rolling a ball back and forth", "Social", "Sitting facing each other and rolling a ball.", { sitting: 0.9, grasp_release: 0.4, joint_attention: 0.4 }, 10, 15, ["turn-taking", "tracking", "release"], "A soft, slow ball; sit close."),
  A("pull_string", "Pull a toy on a string", "Thinking", "Pull a string to bring a toy closer.", { intentional_action: 0.7, cause_effect: 0.6, fine_motor: 0.4 }, 9, 14, ["means-end problem solving"], "Keep strings short and supervise."),
  A("stacking_blocks", "Stacking blocks", "Movement", "Stack 2-3 blocks on top of each other.", { fine_motor: 0.7, spatial_reasoning: 0.4, cause_effect: 0.4, grasp_release: 0.8, hand_eye: 0.6 }, 12, 15, ["fine motor", "spatial reasoning", "patience"], "Large, light blocks are easier; celebrate crashes too."),
  A("nesting_cups", "Nesting cups", "Thinking", "Fitting cups inside one another.", { grasp_release: 0.7, trial_error: 0.4, spatial_reasoning: 0.3 }, 11, 18, ["size ordering", "trial and error"], "Start with two or three cups."),
  A("shape_sorter", "Shape sorter", "Thinking", "Match shapes (circle, square, triangle) to holes.", { shape_matching: 0.6, spatial_reasoning: 0.6, fine_motor: 0.7, trial_error: 0.5 }, 12, 18, ["shape recognition", "problem solving", "fine motor"], "Start with 2-3 shapes; the circle is easiest."),
  A("push_walker", "Push toy walking", "Movement", "Walking while pushing a sturdy wheeled toy.", { standing: 0.8, walking: 0.3 }, 10, 15, ["balance", "walking"], "Choose a stable toy that won't tip."),
  A("simple_puzzle", "Simple puzzle (3-4 pieces)", "Thinking", "Place large knob puzzle pieces in their spots.", { shape_matching: 0.7, fine_motor: 0.7, problem_solving: 0.6, persistence: 0.5 }, 15, 24, ["problem solving", "shape recognition", "persistence"], "Knob puzzles are easier; start with a single piece."),
  A("pretend_play", "Pretend play (feed the doll)", "Social", "Pretending to feed a doll or talk on a toy phone.", { pretend_play: 0.6, action_imitation: 0.8 }, 15, 24, ["symbolic thinking", "language"], "Model the pretend action first."),
];

// ---------------------------------------------------------------- words
// Approximate month by which many babies understand the word (receptive). Rough tiers built from the
// shape of Wordbank / MacArthur-Bates CDI comprehension norms (Frank et al., 2021) and lab evidence for
// early nouns (Bergelson & Swingley, 2012). Individual babies vary by months either way.
const WORDS = {
  7: "mommy mama daddy dada",
  8: "milk banana apple cookie juice spoon nose mouth eyes hair hand feet",
  9: "no bye-bye hi peekaboo bottle",
  10: "ball bath up uh-oh kiss more book",
  11: "dog doggy shoe eat night-night all-gone baby",
  12: "cat kitty car cup hat duck water nap hug come sit outside clap",
  14: "keys sock bird birdie bear phone light give bed diaper teeth tummy ear",
  15: "chair door cow horse sheep flower tree blanket sleep walk push open hot",
  17: "tractor bus train boat pig chicken fish truck grandma grandpa wash help",
  19: "big little wet dirty in on happy sad wait mine",
  22: "under behind tomorrow yesterday red blue yellow green same different",
};
const WORD_AGE = {};
for (const [m, s] of Object.entries(WORDS)) for (const w of s.split(" ")) WORD_AGE[w] = +m;
const VOCAB = { 0: [0, 0], 5: [0, 5], 8: [5, 20], 10: [20, 50], 12: [50, 100], 18: [100, 250], 24: [250, 500] };

// ---------------------------------------------------------------- model checkpoints (phase 2)
const CHECKPOINTS = [0, 3, 6, 12];

return { ACUITY, CONTRAST_SENS, COLOR_SAT, HEAR, DOMAINS, CAPS, level, ACTIVITIES, WORD_AGE, VOCAB, CHECKPOINTS };
})();
