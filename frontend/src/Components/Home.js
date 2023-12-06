import React, { useState } from 'react';
import { Button, Input, Modal } from 'antd';
import axios from 'axios'
const { TextArea } = Input;

const Home = () => {
    const [inputValue, setInputValue] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [text, setText] = useState('');
    const [answer, setAnswer] = useState('')
    const [loading, setLoading] = useState(false);
    const [accuracy, setAccuracy] = useState('');

    const handleOk = () => {
        setIsModalOpen(false);
        window.location.reload();
    };
    const handleCancel = () => {
        setIsModalOpen(false);
        window.location.reload();
    };

    async function onClick() {
        setLoading(true);
        const response = await axios.post("http://localhost:5000/predictDisaster", { description: inputValue })
        console.log(response.data.answer.text);
        setLoading(false);
        setText(response.data.answer.text);
        setAnswer(response.data.answer.Class);
        setAccuracy(response.data.accuracy);

        setIsModalOpen(true);
    }

    function handleInputChange(e) {
        setInputValue(e.target.value);
    }

    return (
        <>
            <TextArea
                rows={4}
                value={inputValue}
                onChange={handleInputChange}
            />
            <Button type='primary' onClick={onClick} loading={loading}>Submit</Button>

            <Modal title="Predicted Result" footer={false} open={isModalOpen} onOk={handleOk} onCancel={handleCancel}>
                <p><strong>Text: </strong>{text}</p>
                <br/>
                <p><strong>Prediction: </strong>{answer}</p>
                <p><strong>Model Accuracy: </strong>{accuracy}</p>
            </Modal>
        </>
    )
}

export default Home;
