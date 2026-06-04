"""Unit tests for parsers/dxf.py — DXF text extraction."""
import os
import tempfile
import pytest

ezdxf = pytest.importorskip("ezdxf", reason="ezdxf not installed")

from parsers.dxf import parse


@pytest.fixture
def dxf_with_text():
    """Create a DXF file with TEXT and MTEXT entities."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text("Hello World", dxfattribs={"insert": (0, 0)})
    msp.add_mtext("Multi-line\ntext here", dxfattribs={"insert": (0, 5)})
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def dxf_with_dimensions():
    """Create a DXF file with DIMENSION entities."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    # Dimension with override text
    dim1 = msp.add_linear_dim(base=(5, 5), p1=(0, 0), p2=(10, 0))
    dim1.dimension.dxf.text = "10.00mm"
    dim1.render()
    # Dimension without override text (actual_measurement via '<>' placeholder)
    dim2 = msp.add_linear_dim(base=(5, 15), p1=(0, 10), p2=(7, 10))
    dim2.render()
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def dxf_with_attributes():
    """Create a DXF file with INSERT + ATTRIB entities."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    block = doc.blocks.new(name="TITLEBLOCK")
    block.add_attdef("TITLE", insert=(0, 0))
    block.add_attdef("AUTHOR", insert=(0, 1))
    insert = msp.add_blockref("TITLEBLOCK", insert=(0, 0))
    insert.add_auto_attribs({"TITLE": "Floor Plan", "AUTHOR": "Architect"})
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def dxf_with_layers():
    """Create a DXF file with custom layers."""
    doc = ezdxf.new()
    doc.layers.add("Electrical", color=1)
    doc.layers.add("Plumbing", color=3)
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def dxf_empty():
    """Create a DXF file with no text entities."""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 10))  # geometry only, no text
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        doc.saveas(f.name)
        yield f.name
    os.unlink(f.name)


class TestDxfTextExtraction:
    def test_text_and_mtext(self, dxf_with_text):
        result = parse(dxf_with_text)
        assert "Hello World" in result
        assert "Multi-line" in result
        assert "## Annotations" in result

    def test_dimension_override_text(self, dxf_with_dimensions):
        result = parse(dxf_with_dimensions)
        assert "10.00mm" in result
        assert "## Dimensions" in result

    def test_dimension_actual_measurement(self, dxf_with_dimensions):
        result = parse(dxf_with_dimensions)
        # The second dimension has no override → uses actual_measurement (7.0)
        assert "7.0" in result

    def test_block_attributes(self, dxf_with_attributes):
        result = parse(dxf_with_attributes)
        assert "TITLE: Floor Plan" in result
        assert "AUTHOR: Architect" in result
        assert "## Attributes" in result


class TestDxfMetadata:
    def test_layers(self, dxf_with_layers):
        result = parse(dxf_with_layers)
        assert "## Layers" in result
        assert "Electrical" in result
        assert "Plumbing" in result

    def test_blocks(self, dxf_with_attributes):
        result = parse(dxf_with_attributes)
        assert "## Blocks" in result
        assert "TITLEBLOCK" in result


class TestDxfStructuredOutput:
    def test_empty_sections_omitted(self, dxf_empty):
        result = parse(dxf_empty)
        assert "## Annotations" not in result
        assert "## Dimensions" not in result

    def test_section_ordering(self, dxf_with_text):
        result = parse(dxf_with_text)
        # Annotations should come before Layers (if both present)
        if "## Layers" in result:
            assert result.index("## Annotations") < result.index("## Layers")


class TestDxfErrorHandling:
    def test_corrupted_file(self, tmp_path):
        bad_file = tmp_path / "corrupted.dxf"
        bad_file.write_text("this is not a dxf file")
        result = parse(str(bad_file))
        assert result == ""

    def test_nonexistent_file(self):
        result = parse("/nonexistent/path.dxf")
        assert result == ""
