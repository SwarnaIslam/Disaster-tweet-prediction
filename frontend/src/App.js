import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Home from './Components/Home'
import LayOut from './Components/LayOut'

const App = () => {
  return (
    <>
      <Routes>
        <Route path="/" element={<LayOut />}></Route>
      </Routes>
    </>
  )
}

export default App