import React from 'react';
import { Layout, Space } from 'antd';
import Home from './Home';
const { Header, Footer, Sider, Content } = Layout;
const headerStyle = {
  textAlign: 'center',
  color: '#fff',
  height: '10vh',
  paddingInline: 50,
  lineHeight: '64px',
//   backgroundColor: '#7dbcea',
};
const contentStyle = {
//   textAlign: 'center',
  minHeight: 120,
  lineHeight: '120px',
  height: '80vh',
  color: '#fff',
  padding: '20px'
//   backgroundColor: '#108ee9',
};
const footerStyle = {
  textAlign: 'center',
  color: '#fff',
//   backgroundColor: '#7dbcea',
  height: '10vh'
};
const LayOut = () => (
  <Space
    direction="vertical"
    style={{
      width: '100%',
    }}
    size={[0, 48]}
  >
    <Layout>
      <Header style={headerStyle}>Disaster Tweet Prediction System</Header>
      <Content style={contentStyle}><Home/></Content>
      <Footer style={footerStyle}></Footer>
    </Layout>
  </Space>
);
export default LayOut;