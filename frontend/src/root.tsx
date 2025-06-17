/**
 * OpenHands 前端应用根组件
 *
 * 技术栈:
 * - React 18 - 前端框架
 * - React Router v7 - 客户端路由
 * - Tailwind CSS - 原子化CSS框架
 * - React Hot Toast - 通知组件
 * - TypeScript - 类型安全
 *
 * 架构说明:
 * 这是OpenHands前端应用的根组件，定义了整个应用的HTML结构和基础布局。
 * 使用React Router的新架构，支持SSR和客户端渲染。
 *
 * 核心功能:
 * 1. HTML文档结构定义
 * 2. 全局样式加载 (Tailwind CSS + 自定义CSS)
 * 3. 元数据管理 (SEO优化)
 * 4. 全局通知系统 (Toast)
 * 5. 路由出口 (Outlet)
 *
 * 设计模式:
 * - 布局组件模式: Layout组件提供通用HTML结构
 * - 插槽模式: 通过children和Outlet提供内容插槽
 * - 元数据模式: 通过meta函数提供SEO信息
 */

// React Router v7 核心组件
import {
  Links,              // 样式和资源链接管理
  Meta,               // 元数据标签管理
  MetaFunction,       // 元数据函数类型
  Outlet,             // 路由出口组件
  Scripts,            // 脚本标签管理
  ScrollRestoration,  // 滚动位置恢复
} from "react-router";

// 全局样式导入
import "./tailwind.css";  // Tailwind CSS框架
import "./index.css";     // 自定义全局样式

import React from "react";
import { Toaster } from "react-hot-toast";  // 全局通知组件

/**
 * 应用布局组件
 *
 * 提供整个应用的HTML文档结构，包括:
 * - HTML头部元数据
 * - 样式和脚本加载
 * - 全局组件 (通知系统)
 * - 滚动位置恢复
 *
 * @param children - 子组件内容
 */
export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* 基础元数据 */}
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />

        {/* 动态元数据 (由各页面的meta函数提供) */}
        <Meta />

        {/* 样式和资源链接 */}
        <Links />
      </head>
      <body>
        {/* 页面内容 */}
        {children}

        {/* 滚动位置恢复 - 在路由切换时恢复滚动位置 */}
        <ScrollRestoration />

        {/* 客户端脚本 */}
        <Scripts />

        {/* 全局通知组件 - 显示成功、错误、警告等消息 */}
        <Toaster />
      </body>
    </html>
  );
}

/**
 * 应用元数据配置
 *
 * 定义应用的SEO信息，包括:
 * - 页面标题
 * - 描述信息
 * - 其他SEO标签
 *
 * 这些信息会被搜索引擎和社交媒体平台使用
 */
export const meta: MetaFunction = () => [
  { title: "OpenHands" },                        // 页面标题
  { name: "description", content: "Let's Start Building!" },  // 页面描述
];

/**
 * 应用根组件
 *
 * 作为React Router的根组件，通过Outlet渲染当前路由对应的页面组件。
 * 这是一个简单的容器组件，实际的页面内容由路由系统管理。
 *
 * 路由流程:
 * URL变化 → Router匹配 → 渲染对应组件到Outlet → 显示页面
 */
export default function App() {
  return <Outlet />;
}
