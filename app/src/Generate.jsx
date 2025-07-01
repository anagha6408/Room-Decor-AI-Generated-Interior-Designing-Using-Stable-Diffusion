import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Form, Button, Image, Spinner, Alert, Card } from 'react-bootstrap';
import './Generate.css';
import defaultPreview from './assets/preview.jpg';

function Generate() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [style, setStyle] = useState('');
  const [outputImage, setOutputImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const styles = ["modern", "bohemian", "industrial", "minimalist"];

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setOutputImage(null);
  };

  const handleStyleChange = (event) => {
    setStyle(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setOutputImage(null);
    setError(null);

    if (!selectedFile || !style) {
      setError("Please select both an image and a style.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('style', style);

    try {
      const response = await axios.post('http://127.0.0.1:5001/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setOutputImage(response.data.image_url);
    } catch (error) {
      setError("Failed to generate image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className='generate-background'>
      <Container className="d-flex justify-content-center align-items-center">
        <Row>
          <Col md={12}>
            <h1 className="text-center main-title">Interior Design Generator</h1>
            <p className="text-muted text-center mb-5">
              Upload an image, select a design style, and generate a custom interior style suggestion.
            </p>
          </Col>
          <Col md={12} className="d-flex justify-content-center align-items-center">
            <Card className="p-4 shadow-sm border-0 container-card">
              <Form onSubmit={handleSubmit}>
                <Row>
                  <Col md={4}>
                    <Form.Group controlId="formFile" className="mb-4">
                      <Form.Label className="fw-semibold">Upload Image</Form.Label>
                      <Form.Control type="file" onChange={handleFileChange} accept="image/*" className="file-input" />
                    </Form.Group>

                    <Form.Group controlId="formStyle" className="mb-4">
                      <Form.Label className="fw-semibold">Select Style</Form.Label>
                      <Form.Control as="select" value={style} onChange={handleStyleChange}>
                        <option value="">Choose a style...</option>
                        {styles.map((styleOption) => (
                          <option key={styleOption} value={styleOption}>
                            {styleOption.charAt(0).toUpperCase() + styleOption.slice(1)}
                          </option>
                        ))}
                      </Form.Control>
                    </Form.Group>

                    <Button
                      variant="primary"
                      type="submit"
                      disabled={loading}
                      className="w-100 generate-button"
                    >
                      {loading ? <Spinner as="span" animation="border" size="sm" /> : 'Generate'}
                    </Button>
                    <Button
                      variant="primary"
                      disabled={loading}
                      className="w-100 generate-button"
                      onClick={() => {
                        if (outputImage) {
                          const link = document.createElement('a');
                          link.href = outputImage;
                          link.download = 'generated_image.png'; // Specify the filename
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);
                        } else {
                          setError("No generated image available to download.");
                        }
                      }}
                    >
                      {loading ? <Spinner as="span" animation="border" size="sm" /> : 'Download'}
                    </Button>


                  </Col>

                  <Col md={8} className="d-flex justify-content-center gap2rem">
                    <div className="text-center mb-4">
                      <h5>Preview</h5>
                      <Image
                        src={selectedFile ? URL.createObjectURL(selectedFile) : defaultPreview}
                        alt="Preview"
                        fluid
                        className={`border ${selectedFile ? 'image-preview' : 'default-preview'}`}
                      />
                    </div>
                    <div className="text-center">
                      <h5>Generated Image</h5>
                      <Image
                        src={outputImage || defaultPreview}
                        alt="Generated"
                        fluid
                        className={`border ${outputImage ? 'image-preview' : 'default-preview'}`}
                      />
                    </div>
                  </Col>
                  <Col md={12}>
                    {error && (
                      <Alert variant="danger" className="mt-4">
                        {error}
                      </Alert>
                    )}
                  </Col>
                </Row>
              </Form>
            </Card>
          </Col>
        </Row>
      </Container>
    </main>
  );
}

export default Generate;
