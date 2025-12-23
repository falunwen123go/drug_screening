from typing import Dict

def check_lipinski(props: Dict[str, float]) -> bool:
    if not props:
        return False
    return (
        props.get('MolecularWeight', 999) <= 500 and
        props.get('LogP', 99) <= 5 and
        props.get('NumHDonors', 99) <= 5 and
        props.get('NumHAcceptors', 99) <= 10
    )
