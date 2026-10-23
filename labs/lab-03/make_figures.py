"""Regenerate the hand-designed lab diagrams using only Python's standard library."""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'notes/assets/lab-03'
OUT.mkdir(parents=True, exist_ok=True)


def text(x, y, value, size=19, color='#172b4d', anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')


def line(x1, y1, x2, y2, color='#64748b', width=3, arrow=False):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"'
            + (' marker-end="url(#arrow)"' if arrow else '') + '/>')


def dot(x, y, color='#126782', radius=7):
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="2"/>'


def rect(x, y, w, h, color='#edf6fa'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" stroke="#64748b" stroke-width="2"/>'


def save(name, height, title, desc, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 {height}" '
           f'role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title>'
           f'<desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="8" '
           'markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" '
           'fill="#64748b"/></marker></defs>'
           f'<rect width="1060" height="{height}" fill="white"/>'
           '<g font-family="Arial, Helvetica, sans-serif">' + ''.join(body) + '</g></svg>\n')
    (OUT / name).write_text(svg)


# A complete one-dimensional example; the exercises use a different 2D mesh.
b = [text(30, 35, 'One mesh, three distinct index systems', 25)]
xs = [230, 410, 590, 770, 950]
b += [text(30, 105, 'Mesh vertices'), line(xs[0], 100, xs[-1], 100)]
for j, x in enumerate(xs[::2]):
    b += [dot(x, 100), text(x, 83, f'vertex {j}', anchor='middle')]
b += [text(410, 133, 'cell A', anchor='middle'), text(770, 133, 'cell B', anchor='middle')]
b += [text(30, 207, 'Local Q2 bases'), line(230, 200, 565, 200), line(615, 200, 950, 200)]
for xx in [(230, 397.5, 565), (615, 782.5, 950)]:
    for x, i in zip(xx, [0, 2, 1]):
        b += [dot(x, 200, '#b45309'), text(x, 235, f'i = {i}', color='#92400e', anchor='middle')]
b += [text(590, 177, 'shared vertex, two local occurrences', 16, anchor='middle')]
b += [text(30, 310, 'Global DoFs'), line(xs[0], 300, xs[-1], 300)]
for x, I in zip(xs, [3, 0, 4, 1, 2]):
    b += [dot(x, 300, '#7c3aed'), text(x, 335, f'I = {I}', color='#6d28d9', anchor='middle')]
b += [text(230, 385, 'Cell A: g_A(0, 1, 2) = (3, 4, 0)'), text(650, 385, 'Cell B: g_B(0, 1, 2) = (4, 2, 1)')]
b += [text(30, 425, 'Illustrative global numbering: the shared DoF has one global index and two local indices.', 18)]
save('numbering-layers.svg', 455, 'Mesh, local basis and global DoF indices',
     'Two 1D Q2 cells. Mesh vertices are 0,1,2. Local bases use endpoint indices 0,1 and midpoint index 2. '
     'The five global DoFs are numbered 3,0,4,1,2 from left to right. The central shared DoF has global index 4.', b)

b = [text(30, 35, 'Degrees of freedom per object: exclusive and inclusive counts', 24)]
b += [rect(80, 90, 260, 260)]
for x in [80, 210, 340]:
    for y in [90, 220, 350]:
        color = '#126782' if x != 210 and y != 220 else '#b45309'
        if x == 210 and y == 220:
            color = '#7c3aed'
        b.append(dot(x, y, color, 9))
b += [text(80, 390, 'A scalar Q2 cell', 20),
      text(430, 100, 'dpo[0]: attached to one vertex', color='#126782'),
      text(430, 155, 'dpo[1]: attached to one edge, excluding endpoints', color='#92400e'),
      text(430, 210, 'dpo[2]: attached to one cell, excluding its boundary', color='#6d28d9'),
      text(430, 280, 'DoFs per face include that face’s endpoint DoFs.'),
      text(430, 320, 'DoFs per cell include all its subobjects.'),
      text(430, 380, '2D: edge = face; quadrilateral = cell.'),
      text(430, 415, '3D: edge, surface face and volume cell are distinct.')]
save('dpo-objects.svg', 450, 'Exclusive dpo versus inclusive face and cell counts',
     'Q2 nodes on a square are colored blue at vertices, orange in edge interiors, and purple in the cell interior. '
     'A face count includes endpoint DoFs; an exclusive edge count does not.', b)

b = [text(30, 35, 'Association with a mesh object is a logical choice', 24)]
for offset, title in [(50, 'Conforming Lagrange'), (580, 'Discontinuous Lagrange')]:
    b.append(text(offset, 83, title, 22))
    b += [rect(offset, 125, 180, 180), rect(offset+180, 125, 180, 180)]
    if offset == 50:
        b += [line(offset+180, 125, offset+180, 305, '#7c3aed', 5),
              dot(offset+180, 215, '#7c3aed', 10),
              text(offset+180, 350, 'same face functional → same global DoF', 17, anchor='middle')]
    else:
        b += [dot(offset+169, 215, '#b45309', 10), dot(offset+191, 215, '#126782', 10),
              text(offset+132, 200, 'I', 22), text(offset+210, 200, 'J', 22),
              text(offset+180, 350, 'cell-local functionals → independent DoFs', 17, anchor='middle')]
b += [text(30, 405, 'DG markers are offset only for readability: two support points can have identical coordinates.', 18),
      text(30, 438, 'Their attachment is to separate cells; comparing point coordinates does not identify the DoFs.', 18)]
save('sharing.svg', 470, 'Shared and cell-local degrees of freedom',
     'Two-cell examples. A conforming face DoF has one shared global index; two discontinuous DoFs can '
     'be evaluated at the same point while retaining independent global indices.', b)

b = [text(30, 35, 'The DoFHandler connects topology and a finite element', 24),
     rect(30, 85, 290, 95), text(175, 123, 'Triangulation', 23, anchor='middle'),
     text(175, 156, 'objects + connectivity + geometry', 17, anchor='middle'),
     rect(30, 220, 290, 95), text(175, 258, 'FiniteElement', 23, anchor='middle'),
     text(175, 291, 'local basis + dual basis + dpo', 17, anchor='middle'),
     line(320, 130, 405, 195, arrow=True), line(320, 265, 405, 205, arrow=True),
     rect(415, 140, 250, 120), text(540, 183, 'DoFHandler', 24, anchor='middle'),
     text(540, 217, 'global DoF numbering', 18, anchor='middle'),
     text(540, 245, 'local-to-global maps g_K', 18, anchor='middle'),
     line(665, 200, 735, 200, arrow=True),
     rect(750, 140, 280, 120), text(890, 183, 'Vector of coefficients', 21, anchor='middle'),
     text(890, 218, 'values U indexed by global DoF', 16, anchor='middle'),
     text(30, 375, 'Renumbering changes global indices and g_K; mesh-object indices and the space stay fixed.', 18),
     text(30, 410, 'To represent the same function, reorder its coefficient vector consistently.', 18)]
save('handler.svg', 445, 'Responsibilities of the mesh, element, DoFHandler and coefficient vector',
     'Triangulation and FiniteElement feed DoFHandler. The coefficient vector is a separate object '
     'indexed by the resulting global DoFs. Renumbering does not renumber mesh objects.', b)

b = [text(30, 35, 'Local object numbers and mesh-object numbers', 25),
     text(30, 80, 'One reference quadrilateral', 21), rect(65, 150, 200, 180)]
for x,y,i in [(65,330,0),(265,330,1),(65,150,2),(265,150,3)]:
    b += [dot(x,y),text(x,y+28 if y==330 else y-16,f'v = {i}',17,anchor='middle')]
for x,y,name in [(34,244,'f = 0'),(302,244,'f = 1'),(165,370,'f = 2'),(165,122,'f = 3')]:
    b.append(text(x,y,name,16,anchor='middle',color='#92400e'))
b += [text(440,80,'A separate three-cell mesh: global object labels',21)]
for c in range(3):
    b += [rect(455+180*c,150,180,180),text(545+180*c,245,f'cell {c}',19,anchor='middle')]
for row,y in enumerate([330,150]):
    for col in range(4):
        x=455+180*col; vertex=4*row+col
        b += [dot(x,y),text(x,y+28 if row==0 else y-16,f'V{vertex}',17,anchor='middle')]
for j in range(3):
    b += [text(545+180*j,309,f'E{j}',17,color='#92400e',anchor='middle'),
          text(545+180*j,179,f'E{j+3}',17,color='#92400e',anchor='middle')]
for j in range(4):
    b.append(text(469+180*j,215,f'E{j+6}',16,color='#92400e'))
b += [text(30,410,'Local vertex/face slots restart on each cell. Mesh-object IDs identify shared objects.',18),
      text(30,443,'These are illustrative topology labels. No global DoF numbering has been assigned here.',18)]
save('mesh-entities.svg',475,'Local subobject indices versus mesh-object indices',
     'Left: reference quad local vertex slots 0 to 3 and face slots 0 to 3. '
     'Right: a separate three-cell mesh with global vertex labels V0 to V7 '
     'and edge labels E0 to E9. These are topology labels, not DoF labels.',b)
