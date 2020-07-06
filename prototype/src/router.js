import Vue from 'vue'
import VueRouter from 'vue-router'
import ActiveLearning from './components/ActiveLearning'
import Playground from './components/Playground'

Vue.use(VueRouter)

export default new VueRouter({
    routes: [{
        path: '/',
        component: ActiveLearning
    },
    {
        path: "/playground",
        component: Playground
    }],
    mode: 'history'
})