import re
import pandas as pd

def clean_skill_list(skills):
    """
    [HINGLISH]
    Use: Ye function `views.py` me resume upload ke time aur `services.py` me use hota h.
    Why: Raw skills jo resume se aate h wo gande hote h (duplicates, formatting issues), unhe saaf karne ke liye banaya h.
    Effect: Isse duplicate skills hat jaate h aur sab ek format me aa jate h, jisse matching accurate hoti h.
    """
    try:
        keep_single_letter = {'c', 'r'}
        acronyms = {'SQL', 'HTML', 'CSS', 'REST', 'AWS', 'GCP', 'NLP', 'ETL', 'API', 'CI/CD', 'C++', 'C#', 'R', 'AI', 'IOT'}
        cleaned = []
        seen = set()
        for s in skills:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if not s:
                continue
            
            s = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\-]+', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()

            sl = s.lower()
            if len(sl) < 2 and sl not in keep_single_letter:
                continue
            if len(sl.split()) > 4:
                continue
            if re.fullmatch(r'[\W_]+', sl):
                continue

            # Title case, then restore acronyms and special tokens
            nice = s.title()
            tokens = []
            for token in re.split(r'(\/|\+\+|#)', nice):
                up = token.upper().strip()
                if up in acronyms or token in ('/','++', '#'):
                    tokens.append(up)
                elif token.strip():
                    tokens.append(token.strip())
            nice = ' '.join(tokens).replace(' / ', '/').replace('Ci/Cd', 'CI/CD').replace('C++', 'C++').replace('C#', 'C#')
            
            key = nice.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(nice)
        return cleaned
    except Exception:
        return list(dict.fromkeys([str(s).strip() for s in skills if str(s).strip()]))

def clean_title_for_merge(title):
    """
    [HINGLISH]
    Use: Ye `services.py` me career matching aur market data merge karte waqt use hota h.
    Why: Alag-alag jagah career names alag ho sakte h (e.g., "Software Engineer" vs "software engineer"), unhe match karne ke liye normalize karna padta h.
    Effect: Isse duplicates remove hote h aur data sahi se merge hota h.
    """
    if pd.isna(title) or not isinstance(title, str):
        return ""
    text = title.lower().replace('"', '').replace("'", '').strip()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smart_split_skills(text):
    """
    [HINGLISH]
    Use: Ye `services.py` me required skills ko CSV se read karte time use hota h.
    Why: Kabhi kabhi skills me brackets hote h jaise "Java (Basic, Advanced)", normal split se ye toot jata h. Ye function brackets ka dhyan rakhta h.
    Effect: Skills sahi se extract hote h, adhe-adhure nahi aate.
    """
    if not text:
        return []

    skills = []
    current = []
    depth_round = 0    # ()
    depth_square = 0  # []
    depth_curly = 0   # {}

    for char in text:
        if char == '(':
            depth_round += 1
        elif char == ')':
            depth_round -= 1
        elif char == '[':
            depth_square += 1
        elif char == ']':
            depth_square -= 1
        elif char == '{':
            depth_curly += 1
        elif char == '}':
            depth_curly -= 1

        # COMMA OUTSIDE bracket = SPLIT
        if char == ',' and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            skill = ''.join(current).strip()
            if skill:
                skills.append(skill)
            current = []
        else:
            current.append(char)

    # Add last item
    last = ''.join(current).strip()
    if last:
        skills.append(last)

    return skills

def format_salary(salary_value):
    """
    [HINGLISH]
    Use: Ye `services.py` me market data dikhane ke liye use hota h.
    Why: Salary numbers ko readable format (e.g., 120,000) me dikhane ke liye.
    Effect: User ko salary dekhne me aasani hoti h, UI accha lagta h.
    """
    try:
        if pd.isna(salary_value) or salary_value == 0 or salary_value == '0':
            return get_default_salary()
        
        salary_num = float(salary_value)
        if salary_num >= 1000:
            return f"{salary_num:,.0f}"
        else:
            return f"{salary_num:.0f}"
    except (ValueError, TypeError):
        return get_default_salary()

def get_default_salary():
    """
    [HINGLISH]
    Use: Ye `format_salary` me fallback ke liye use hota h.
    Why: Agar salary data missing h to 0 dikhane se accha ek default value dikhana.
    Effect: Empty ya 0 salary nahi dikhti, user experience maintain rehta h.
    """
    return "85,000"

def get_default_salary_by_title(title):
    """
    [HINGLISH]
    Use: Ye `services.py` me tab use hota h jab CSV me salary na mile.
    Why: Har career ki salary alag hoti h, to title ke hisab se ek realistic guess lagane ke liye banaya h.
    Effect: Data missing hone par bhi user ko relevant salary range dikhti h.
    """
    title_lower = title.lower()
    
    salary_ranges = {
        'software': "120,000",
        'developer': "110,000", 
        'engineer': "115,000",
        'data': "105,000",
        'scientist': "120,000",
        'analyst': "75,000",
        'manager': "95,000",
        'designer': "85,000",
        'marketing': "70,000",
        'sales': "65,000",
        'financial': "80,000",
        'consultant': "90,000"
    }
    
    for key, salary in salary_ranges.items():
        if key in title_lower:
            return salary
    
    return "85,000"

def get_default_growth_by_title(title):
    """
    [HINGLISH]
    Use: Ye `services.py` me growth rate missing hone par use hota h.
    Why: Tech jobs ki growth alag hoti h aur admin jobs ki alag, isliye title se guess karte h.
    Effect: Trends page par growth rate kabhi khali nahi dikhta.
    """
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['data', 'ai', 'machine learning', 'software']):
        return 15.5
    elif any(word in title_lower for word in ['developer', 'engineer', 'cloud']):
        return 12.5
    elif any(word in title_lower for word in ['analyst', 'consultant']):
        return 9.5
    else:
        return 7.5

def calculate_demand_level(growth_rate):
    """
    [HINGLISH]
    Use: Ye `services.py` me demand level (High/Medium/Low) batane ke liye use hota h.
    Why: Sirf percentage dekh kar user ko samajh nahi aata, isliye usse High/Medium me convert kiya h.
    Effect: User ko turant pata chalta h ki job market kaisi h.
    """
    if not growth_rate or pd.isna(growth_rate):
        return 'Medium'
    
    try:
        growth = float(growth_rate)
        if growth > 0.20: 
            return 'High'
        elif growth > 0.10:
            return 'Medium'
        else:
            return 'Low'
    except (ValueError, TypeError):
        return 'Medium'

def generate_demand_level_by_title(title):
    """
    
    Use: Ye `services.py` me tab use hota h jab growth rate ka data hi na ho.
    Why: Job title se hum guess kar sakte h ki demand kaisi h (e.g., AI is High).
    Effect: Data na hone par bhi user ko 'High Demand' ya 'Medium Demand' dikhta h.
    """
    title_lower = title.lower()
    
    # High demand careers
    high_demand_keywords = [
        'software', 'developer', 'engineer', 'data scientist', 'ai', 
        'machine learning', 'cybersecurity', 'cloud', 'devops'
    ]
    
    # Medium demand careers  
    medium_demand_keywords = [
        'analyst', 'consultant', 'manager', 'designer', 'marketing',
        'product', 'project', 'business'
    ]
    
    for keyword in high_demand_keywords:
        if keyword in title_lower:
            return 'High'
    
    for keyword in medium_demand_keywords:
        if keyword in title_lower:
            return 'Medium'
    
    return 'Low'

def safe_load_df(filename):
    """
    [HINGLISH]
    Use: Ye `services.py` me CSV files load karne ke liye use hota h.
    Why: Agar file missing ho ya corrupt ho to code crash na kare, aur columns ko normalize (lowercase) karne ke liye.
    Effect: Code robust banta h, file na hone par bhi site chalti rehti h.
    """
    try:
        # Use on_bad_lines='skip' to handle malformed rows (e.g. extra commas)
        df = pd.read_csv(f'datasets/{filename}', on_bad_lines='skip')
        # CRITICAL FIX: Normalize column names (e.g., 'Skill Name' -> 'skill_name')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# --- SKILL NORMALIZATION ---

SKILL_ALIASES = {
    # C / C++ Family
    'c': 'c programming',
    'programming language c': 'c programming',
    'c language': 'c programming',
    'cpp': 'c++',
    'c plus plus': 'c++',
    'vc++': 'c++',
    'c#': 'c sharp',
    'c sharp': 'c sharp',
    
    # Microsoft Office
    'ms office': 'microsoft office',
    'office 365': 'microsoft office',
    'msoffice': 'microsoft office',
    'powerpoint': 'microsoft powerpoint',
    'ppt': 'microsoft powerpoint',
    'excel': 'microsoft excel',
    'word': 'microsoft word',
    
    'mongo': 'mongodb',
}

def normalize_skill(skill_name):
    """
    [HINGLISH]
    Use: Ye function skill names ko standard format me convert karta h.
    Why: Taaki 'C' aur 'Programming Language C' same maane jaaye.
    Effect: Skill matching accurate ho jaati h.
    """
    if not skill_name or not isinstance(skill_name, str):
        return ""
    
    # 1. Basic cleaning
    clean = skill_name.lower().strip()
    
    # 2. Check aliases
    if clean in SKILL_ALIASES:
        return SKILL_ALIASES[clean]
        
    # 3. Handle special cases (remove version numbers, etc. if needed)
    # For now, just return the cleaned string
    return clean

# --- LEARNING RESOURCES DATA ---

# Static data removed as per dynamic upgrade.

