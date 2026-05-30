# CLINICAL TECHNICAL REPORT: SOFTPREDICT AI PIPELINE & ALGORITHMS

**Document Version:** 1.1.0  
**Prepared For:** Maxillofacial & Orthodontic Clinical Integration  
**Description:** Technical architectural flow and algorithmic breakdown of the SoftPredict AI facial simulation system.

---

## 1. Executive Summary
SoftPredict AI is a clinical-grade orthodontic and orthognathic simulation engine. It predicts how a patient's soft-tissue facial structures (lips, cheeks, chin, and jawline) will adjust following dental alignment or skeletal surgery. 

The system leverages a **Graph Convolutional Network (GCN)** for coordinate regression, enforces biomechanical boundaries to protect critical facial anchors, and uses a local **Generative Facial Prior GAN (GFPGAN)** to reconstruct razor-sharp, realistic skin, lip, and teeth textures—completely eliminating post-surgical simulation blurriness while preserving 100% of the patient's identity.

---

## 2. System Architecture Flow
The workflow consists of an edge-to-cloud-to-edge pipeline, shifting from user interaction to deep learning inference, image processing, and back to presentation.

### Architectural Diagram
```
[ Patient Photo Uploaded in Flutter App ]
                   │
                   ▼ (HTTP POST Multipart Form-Data)
[ FastAPI Backend: Entry Endpoint ]
                   │
                   ▼
[ Step 1: MediaPipe 3D Landmark Mesh Extraction ] (468 Coordinates)
                   │
                   ▼
[ Step 2: GCN Predicts Geometry Shift Vector ] (Coord Regression)
                   │
                   ▼
[ Step 3: Biomechanical Boundary Enforcement ] (Stable Anchors & Outer Framing)
                   │
                   ▼
[ Step 4: Triangulation & Affine Image Warping ] (Geometric Reshaping)
                   │
                   ▼
[ Step 5: Generative AI Face Restoration (GFPGAN) ] (Texture Enhancement & De-blurring)
                   │
                   ▼
[ Step 6: Soft Blending & Color Balancing ] (Post-Processing & Matplotlib 3D Render)
                   │
                   ▼ (JSON URLs payload)
[ Results Displayed in Dashboard Carousel ] (Before, Mesh, After, 3D Graph)
```

---

## 3. Step-by-Step Execution Pipeline

### Step 3.1: Patient Intake & Upload
*   **Action**: The clinician registers a new patient in the Flutter app, filling out clinical metadata (MRN, DOB, Problem, and Method: Teeth vs. Jaw).
*   **Transfer**: The photo is uploaded as raw binary bytes over a secure, connection-closing HTTP request to ensure compatibility with mobile networks.

### Step 3.2: 3D Facial Landmark Extraction
*   **Action**: The backend processes the image using MediaPipe Face Mesh.
*   **Technical Details**: The system registers 468 discrete 3D spatial points. If a patient turns their head slightly, the coordinates adjust dynamically, mapping out eyes, nose, cheeks, outer jawline, and lip contours.

### Step 3.3: Graph Convolutional Network (GCN) Prediction
*   **Action**: The backend inputs the 468 coordinates into the GCN model.
*   **Technical Details**:
    *   Unlike traditional neural networks that treat images as flat grids, the GCN treats the face as a **social network of coordinates** (a Graph).
    *   It passes messages along the edges connecting neighboring landmarks.
    *   Based on prior clinical dataset training, it predicts a displacement vector (shift direction and distance) for each point.

### Step 3.4: Biomechanical Constraint Enforcement
*   **Action**: The system refines the GCN's predicted shifts to ensure clinical safety.
*   **Technical Details**:
    *   **Stable Zones**: Shift limits are set to zero for coordinates around the eyes, upper nose bridge, and forehead, preventing unrealistic distortions.
    *   **Outer Anchors**: Hairline, ears, and neck coordinates are locked to ensure the background remains stable.
    *   **Isolation**: Only the lower jawline and the lip boundary points are allowed to slide along the predicted biomechanical path.

### Step 3.5: Piecewise Affine Warping
*   **Action**: The geometry of the patient's face is reshaped.
*   **Technical Details**:
    *   The 468 points are grouped into thousands of tiny adjacent triangles (Delaunay Triangulation).
    *   The pixels inside each triangle of the original photo are warped and stretched to match the new, predicted positions.

### Step 3.6: Generative AI Face Restoration (GFPGAN)
*   **Action**: Stretched and blurry regions are repaired.
*   **Technical Details**:
    *   Warping stretches lips, teeth, and skin, causing blurriness.
    *   **GFPGAN (Generative Facial Prior GAN)** acts as a texture restorer. It analyzes the warped image, detects the facial layout, and uses an internal StyleGAN generator to redraw high-resolution teeth alignment, sharp lip contours, and clean skin pores.
    *   Importantly, it merges this texture back onto the GCN-warped shape, ensuring the final image is **sharp, realistic, and matches the patient's actual face structure**.

### Step 3.7: Blending & Presentation
*   **Action**: The restored face is blended back into the original photo.
*   **Technical Details**:
    *   A soft feathering mask is applied around the restored face boundary.
    *   The ears, neck, and hair from the original photo are blended with the restored face to eliminate sharp lines.
    *   The FastAPI backend generates a 3D matplotlib wireframe graph of the shifts and returns the complete asset bundle to the Flutter app.

---

## 4. Key Algorithm Descriptions

### 1. Graph Convolutional Network (GCN)
Traditional convolutional layers only analyze rectangular grids (pixels). A GCN operates directly on graph networks. By looking at the distance and angle relationships between facial landmarks (edges), the GCN learns how the movement of one muscle group (e.g., mandible advancement) affects neighboring soft tissue areas (e.g., lower lip projection).

### 2. Piecewise Affine Warping
By dividing the face into a mesh of triangles, this algorithm warps the image locally. Instead of stretching the whole photo, it shifts only the exact pixels between landmarks. For example, when simulating jaw surgery, only the chin and lower jaw pixels are transformed, while the nose and eyes remain completely unaffected.

### 3. Generative Facial Prior (GFP) GAN
GFPGAN uses a rich dictionary of facial features (like eyes, teeth, and lips) learned from millions of high-resolution images. When it encounters a blurry, warped mouth region, it uses these facial priors to reconstruct the natural, clean geometry of the teeth and lips. It acts as an intelligent, context-aware upscaler specifically optimized for human faces.
