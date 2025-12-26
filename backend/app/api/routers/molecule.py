from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

router = APIRouter()

@router.get("/molecule/image")
async def get_molecule_image(
    smiles: str = Query(..., description="SMILES string of the molecule"),
    width: int = Query(300, description="Image width"),
    height: int = Query(300, description="Image height")
):
    """
    Generate a 2D molecular structure image from SMILES string.
    Returns PNG image.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Generate image
        img = Draw.MolToImage(mol, size=(width, height))
        
        # Convert to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(content=img_bytes.getvalue(), media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate molecule image: {str(e)}")
